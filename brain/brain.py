"""
智能体大脑 v2.2 — 四状态机 + MCP 工具池 + 三层记忆

状态流转: THINK → EXECUTE ⇄ REFLECT → END
"""

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from json_utils import parse_llm_json
from tools.mcp_pool import MCPToolPool
from memory import MemoryManager
from memory.models import Message, Function as MsgFunction, ToolCall as MsgToolCall
from memory.models import filter_orphan_tool_messages

from .adapters import BaseModelAdapter, ModelConfig, create_model_adapter
from .state import LoopMode, LoopState


# ==================== 兼容层 ====================

class ToolRegistry:
    """兼容旧 API 的注册表，委托给 MCPToolPool"""

    def __init__(self, pool: MCPToolPool):
        self._pool = pool

    def register(self, name: str, description: str, function: Callable, parameters: Optional[Dict] = None):
        self._pool.add_live_tool(name, description, function, parameters)

    def create_tool_dynamically(self, name: str, code: str, description: str = "") -> bool:
        result = self._pool.register_tool(name, description, code)
        return "成功" in result

    def get_tool(self, name: str) -> Optional[Dict]:
        return self._pool.get_tool(name)

    def list_tools(self) -> List[Dict]:
        return self._pool.list_tools()

    def execute(self, name: str, arguments: Dict) -> str:
        return self._pool.execute(name, arguments)


# ==================== 智能体大脑 ====================

class AgentBrain:
    """
    智能体大脑 v2.2 — 四状态机

    THINK  → 分析任务、制定计划、判断是否需要工具
    EXECUTE → 调用工具 / 创建工具 / 完成任务
    REFLECT → 错误分析、寻找解决方案、修正计划
    END    → 输出最终答案
    """

    def __init__(
        self,
        model_configs: List[ModelConfig],
        tools: Optional[ToolRegistry] = None,
        tool_pool: Optional[MCPToolPool] = None,
        system_prompt: str = "你是一个有用的AI助手，擅长规划和使用工具完成任务。",
        max_iterations: int = 10,
        max_retries: int = 3,
        short_memory_path: str = "./memory/archives/conversation_memory.json",
        long_memory_path: str = "./memory/archives/long_term_archive.json",
        openai_client: Any = None,
        memory: Optional[MemoryManager] = None,
    ):
        # 模型
        self.models = [create_model_adapter(c) for c in model_configs]
        self.current_model_idx = 0
        self.think_model_idx = 0

        # MCP 工具池
        self.tool_pool = tool_pool or MCPToolPool(
            pool_file="./tools/mcp_tools.json", code_dir="./tools/tool_add/tool_direct")
        self.tools = tools or ToolRegistry(self.tool_pool)

        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_retries = max_retries

        # 内置工具保护列表（不会被自动清理）
        self.PROTECTED_TOOLS = [
            "read_file", "write_file", "create_file", "delete_file",
            "create_directory", "run_command", "delete_tool",
            # 脑管理工具（live tools，输出含"错误"等词会被误判）
            "switch_model", "list_models", "get_status",
            "get_memory_summary", "search_long_memory",
            "list_mcp_tools", "search_mcp_tools",
        ]

        # 三层记忆：可外部注入或自建
        self.openai_client = openai_client
        if memory:
            self.memory = memory
        else:
            self.memory = MemoryManager(
                client=openai_client,
                model=self.models[0].get_model_name() if self.models else "gpt-4o",
                working_memory_file=short_memory_path,
                long_term_memory_file=long_memory_path,
            )

        self.state = LoopState(max_iterations=max_iterations, max_retries=max_retries)
        self._breakpoint: Optional[Dict] = None  # 任务超限终止时保存的断点
        self._register_brain_tools()

    # ==================== 模型管理 ====================

    @property
    def current_model(self) -> BaseModelAdapter:
        return self.models[self.current_model_idx]

    @property
    def think_model(self) -> BaseModelAdapter:
        return self.models[self.think_model_idx]

    def switch_model(self, index: int = None):
        if index is None:
            self.current_model_idx = (self.current_model_idx + 1) % len(self.models)
        else:
            self.current_model_idx = index % len(self.models)
        print(f"🔄 切换到模型: {self.current_model.get_model_name()}")

    # ==================== 上下文构建 ====================

    # 纯信息查询类工具——调用它们不算"完成任务"
    _INFO_TOOLS = {
        "list_mcp_tools", "search_mcp_tools", "get_status",
        "get_memory_summary", "search_long_memory", "list_models",
        "switch_model",
    }

    # 内部元提示词前缀（我们注入的指令，非用户真实输入，应过滤）
    _META_PROMPT_PREFIXES = (
        "用户说:", "任务执行出错", "执行计划。调用工具",
        "数据已获取，请用write_file", "任务完成。请用中文",
        "你是任务分析器", "判断下一步",
    )

    # 内部元消息模式（LLM 返回的 think/reflect JSON，对后续对话无意义）
    _META_JSON_KEYS = ("done", "root_cause", "fix", "can_retry", "already_have_data")

    @classmethod
    def _is_meta_message(cls, msg: Dict) -> bool:
        """判断消息是否为内部元消息（非用户可见对话）"""
        role = msg.get("role", "")
        content = msg.get("content", "") or ""

        # 永不过滤带 tool_calls 的消息——拆散配对会导致 API 400
        if msg.get("tool_calls"):
            return False

        # 我们注入的指令提示词
        if role == "user" and content.startswith(cls._META_PROMPT_PREFIXES):
            return True

        # LLM 返回的 think/reflect JSON（含 done/root_cause/fix 等内部标志）
        if role == "assistant" and content:
            stripped = content.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    data = json.loads(stripped)
                    if any(k in data for k in cls._META_JSON_KEYS):
                        return True
                except json.JSONDecodeError:
                    pass

        return False

    def _messages_to_dict_list(self, messages: List, clean_meta: bool = False) -> List[Dict]:
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get('content') is None and msg.get('tool_calls'):
                    msg = {**msg, 'content': '[tool_call]'}
                result.append(msg)
            elif hasattr(msg, 'to_dict'):
                d = msg.to_dict()
                if d.get('content') is None and d.get('tool_calls'):
                    d = {**d, 'content': '[tool_call]'}
                result.append(d)

        # 过滤孤立的 tool 消息
        result = filter_orphan_tool_messages(result)

        # 过滤内部元消息（think/reflect JSON + 注入的指令提示词）
        if clean_meta:
            result = [m for m in result if not self._is_meta_message(m)]

        return result

    def _register_brain_tools(self):
        """注册大脑专属工具到 MCP 池"""

        def switch_model_tool(model_index: int = None):
            self.switch_model(model_index)
            return f"已切换到模型: {self.current_model.get_model_name()}"

        def list_models():
            return "\n".join(f"{i}: {m.get_model_name()}" for i, m in enumerate(self.models))

        def get_status():
            return (f"迭代: {self.state.iteration}/{self.max_iterations}, "
                    f"模式: {self.state.mode.value}, "
                    f"错误: {self.state.error_count}次, "
                    f"模型: {self.current_model.get_model_name()}, "
                    f"MCP工具: {self.tool_pool.get_tool_count()['total']}")

        def get_memory_summary():
            return self.memory.summary()

        def search_long_memory(keyword: str):
            results = self.memory.recall(keyword, top_k=5, include_working=False)
            if results:
                lines = [f"找到 {len(results)} 条相关记忆:"]
                for r in results:
                    lines.append(f"- [{r.get('source', '?')}] {r.get('content', '')[:100]}")
                return "\n".join(lines)
            return f"未找到与 '{keyword}' 相关的记忆"

        def list_mcp_tools():
            return self.tool_pool.summary()

        def search_mcp_tools(query: str):
            results = self.tool_pool.search(query)
            if results:
                return f"找到 {len(results)} 个匹配:\n" + "\n".join(
                    f"- {r['name']}: {r['description']}" for r in results)
            return f"未找到与 '{query}' 匹配的工具，可使用 create_tool 创建"

        self.tool_pool.add_live_tool("switch_model", "切换到另一个可用模型", switch_model_tool)
        self.tool_pool.add_live_tool("list_models", "列出所有可用模型", list_models)
        self.tool_pool.add_live_tool("get_status", "获取当前智能体运行状态", get_status)
        self.tool_pool.add_live_tool("get_memory_summary", "获取对话记忆统计", get_memory_summary)
        self.tool_pool.add_live_tool("search_long_memory", "搜索记忆", search_long_memory)
        self.tool_pool.add_live_tool("list_mcp_tools", "列出MCP池中所有工具", list_mcp_tools)
        self.tool_pool.add_live_tool("search_mcp_tools", "搜索MCP工具池", search_mcp_tools)

    # ==================== 重置 ====================

    def reset(self, task: str = None):
        """重置状态，注入系统提示和记忆上下文"""
        memory_ctx = self.memory.get_context_for_llm()
        full_prompt = self.system_prompt
        if memory_ctx:
            full_prompt += "\n\n" + memory_ctx

        self.memory.set_system_message(full_prompt)

        self.state = LoopState(max_iterations=self.max_iterations,
                               max_retries=self.max_retries)
        if task:
            self.state.task = task

    # ==================== 辅助方法 ====================

    # 压缩规则块 — 每次 LLM 调用前注入，防止长上下文遗忘系统提示词
    _RULES_MARKER = "【工作流】"
    _RULES_BLOCK = (
        "【工作流】已有工具能完成→直接调用 | 不能→create_tool创建 | "
        "创建后立即调用获取数据 | 出错→修正参数/换方案 | run_command仅pip/版本检查"
    )

    def _inject_rules(self, messages: List[Dict]) -> List[Dict]:
        """在消息列表前注入压缩系统规则，防止长上下文遗忘"""
        # 只在消息超过一定长度时注入（避免短对话冗余）
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars < 2000:
            return messages
        # 检查是否已有规则注入（避免重复）——规则在头部 system 消息，需查全部 system
        for m in messages:
            if (isinstance(m, dict) and m.get("role") == "system"
                    and self._RULES_MARKER in str(m.get("content", ""))):
                return messages
        return [{"role": "system", "content": self._RULES_BLOCK}] + list(messages)

    def _call_llm(self, messages: List[Dict], tools: Optional[List] = None,
                  use_think_model: bool = True, json_mode: bool = False) -> Dict:
        """统一 LLM 调用入口，自动注入压缩规则防止遗忘"""
        model = self.think_model if use_think_model else self.current_model
        msgs = self._inject_rules(messages)
        try:
            return model.chat(msgs, tools, json_mode=json_mode)
        except TypeError:
            # 兼容旧签名的自定义 adapter（不支持 json_mode 参数）
            return model.chat(msgs, tools)

    # JSON 解析连续失败熔断阈值
    MAX_PARSE_FAILURES = 3
    # 同签名工具调用熔断阈值
    MAX_REPEAT_CALLS = 3
    # 连续无进展 THINK 往返熔断阈值
    MAX_NO_PROGRESS = 4

    def _call_llm_json(self, messages: List[Dict],
                       expected_keys: Tuple[str, ...] = (),
                       use_think_model: bool = True) -> Optional[Dict]:
        """LLM JSON 调用：json_mode + 健壮解析 + 失败纠错重试 1 次。

        仍失败返回 None 并累计 state.parse_failures（供熔断判断）；
        成功则清零计数。
        """
        response = self._call_llm(messages, None,
                                  use_think_model=use_think_model, json_mode=True)
        raw = response.get("content")
        data = parse_llm_json(raw, expected_keys)
        if data is None:
            # 带纠错提示重试 1 次
            retry_messages = list(messages) + [
                {"role": "assistant", "content": str(raw or "")[:500]},
                {"role": "user", "content": (
                    "你上次的输出不是合法的 JSON。"
                    "请只输出一个 JSON 对象，不要 markdown 围栏，不要任何解释文字。"
                )},
            ]
            response = self._call_llm(retry_messages, None,
                                      use_think_model=use_think_model, json_mode=True)
            data = parse_llm_json(response.get("content"), expected_keys)

        if data is None:
            self.state.parse_failures += 1
        else:
            self.state.parse_failures = 0
        return data

    def _abort_to_end(self, reason: str) -> LoopMode:
        """熔断兜底：保存断点 + 直接生成中止回复（不再调 LLM）→ END。

        放在状态函数内部调用，process_turn 和 resume 两条主循环均受保护。
        """
        self._breakpoint = {
            "task": self.state.task,
            "plan": self.state.plan,
            "history": list(self.state.history),
            "error_count": self.state.error_count,
            "iteration": self.state.iteration,
            "abort_reason": reason,
        }
        answer = f"⚠️ 任务中止：{reason}。已保存断点，输入「继续」或 /continue 可恢复。"
        self.state.final_answer = answer
        self.memory.add_assistant_message(answer)
        print(f"  🛑 {answer}")
        return LoopMode.END

    @staticmethod
    def _clean_xml_artifacts(text: str) -> str:
        """移除 LLM 可能泄露的 XML/函数调用语法"""
        # 配对标签（如 <function_calls>...</function_calls>）
        text = re.sub(
            r'<\s*(function_calls|invoke|parameter|xml)[^>]*>.*?<\s*/\s*\1[^>]*>',
            '', text, flags=re.DOTALL)
        # 单独残留的空标签（如 <invoke name="..."> 无闭合，或 </invoke> 无开头）
        text = re.sub(
            r'<\s*/?\s*(function_calls|invoke|parameter|tool_calls?)[^>]*>',
            '', text, flags=re.DOTALL)
        return text.strip()

    def _generate_final_answer(self, original_task: str) -> str:
        """生成友好的中文最终回复，并调用 remember_turn"""
        ctx = self._messages_to_dict_list(self.memory.working_memory.messages, clean_meta=True)
        ctx.append({"role": "user", "content": (
            f"任务完成。请用中文给用户一个简洁的最终回复。"
            f"不要输出任何XML/JSON/工具调用语法。原任务: {original_task}"
        )})
        response = self.current_model.chat(ctx, None)  # 不传工具，纯文本回答
        answer = response.get("content", "") or "完成"
        answer = self._clean_xml_artifacts(answer)
        self.memory.add_assistant_message(answer)
        self.memory.remember_turn(original_task, answer)
        return answer

    def _build_tools_brief(self) -> str:
        """构建可用工具简要列表（含参数签名），供 THINK 决策用"""
        names = list(self.tool_pool.tools.keys())
        if not names:
            return "(仅有 create_tool 元工具)"
        briefs = []
        for name in sorted(names)[:12]:
            tool = self.tool_pool.tools.get(name, {})
            desc = tool.get("description", "")[:60]
            # 提取参数签名
            params = tool.get("parameters", {}).get("properties", {})
            if params:
                param_str = ", ".join(
                    f"{k}:{v.get('type','str')}" for k, v in params.items())
                briefs.append(f"  {name}({param_str}) — {desc}")
            else:
                briefs.append(f"  {name}() — {desc}")
        extra = f"\n  ...共{len(names)}个" if len(names) > 12 else ""
        return "\n".join(briefs) + extra

    def _build_history_brief(self) -> str:
        """构建已执行操作的简要摘要"""
        if not self.state.history:
            return ""
        briefs = []
        for h in self.state.history[-10:]:
            mode = h.get("mode", "")
            if mode == "execute":
                tool = h.get("tool", "?")
                result = str(h.get("result", ""))[:80]
                briefs.append(f"[执行] {tool} → {result}")
            elif mode == "reflect":
                briefs.append(f"[反思] {str(h.get('reflection', ''))[:60]}")
            elif mode == "think":
                briefs.append(f"[思考] {str(h.get('thought', ''))[:60]}")
        return "\n".join(briefs) if briefs else ""

    def _make_assistant_tool_msg(self, response: Dict, tool_calls: List[Dict]) -> Message:
        """构建带 tool_calls 的 assistant 消息（DeepSeek 要求）"""
        tcs = [MsgToolCall(
            id=tc.get("id", ""),
            type=tc.get("type", "function"),
            function=MsgFunction(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            )
        ) for tc in tool_calls]
        return Message(role="assistant", content=response.get("content") or "[tool_call]",
                       tool_calls=tcs)

    # 工具失败的前缀约定（mcp_pool 内部错误均以这些前缀开头）
    _ERROR_PREFIXES = ("错误", "失败", "安全警告", "🚫", "[Ollama 错误",
                       "执行工具", "读取文件失败", "写入文件失败", "创建文件失败",
                       "删除文件失败", "创建目录失败", "执行命令出错")

    @classmethod
    def _is_error_result(cls, result: str) -> bool:
        """前缀匹配判断工具结果是否为错误（避免文件内容含'错误'二字被误判）"""
        return str(result).lstrip().startswith(cls._ERROR_PREFIXES)

    def _execute_single_tool_call(self, func_name: str, args: Dict,
                                   tool_call_id: str = "") -> Tuple[bool, str]:
        """执行单个工具调用，返回 (is_error, result)。含自动清理逻辑。"""
        result = self.tool_pool.execute(func_name, args)

        # 自动清理：create_tool 成功 → 删除同前缀旧版本
        if func_name == "create_tool" and "成功" in result:
            new_name = args.get("name", "")
            prefix = new_name.rstrip('0123456789_v')
            for old in list(self.tool_pool.tools.keys()):
                if (old != new_name and old.startswith(prefix)
                        and old not in self.PROTECTED_TOOLS
                        and len(old) >= len(prefix)
                        and old[:len(prefix)] == prefix):
                    self.tool_pool.delete_tool(old)
                    print(f"  🗑️ 清理旧版: {old}")
            print(f"  🆕 新工具入池: {new_name}")

        # 判断是否出错：前缀约定匹配（用于状态机流转）
        is_error = self._is_error_result(result)

        return is_error, result

    # ==================== 四状态机 ====================

    def _think(self) -> LoopMode:
        """THINK: 多轮思考（最多 max_think_rounds 轮）→ EXECUTE 或 END"""
        self.state.think_rounds = 0
        done = False
        plan = ""
        plan_data: Dict = {}

        while self.state.think_rounds < self.state.max_think_rounds and not done:
            self.state.think_rounds += 1
            history_brief = self._build_history_brief()

            think_prompt = (
                f"任务: {self.state.task}\n"
                f"轮次: {self.state.iteration} | 错误: {self.state.error_count}/{self.state.max_retries}\n"
                f"📦 已有工具:\n{self._build_tools_brief()}\n\n"
                f"决策流程:\n"
                f"1. 池中已有工具能完成任务? → plan=调用工具名, done=false\n"
                f"2. 池中无对应工具? → plan=create_tool 创建, done=false\n"
                f"3. 已成功获取数据? → done=true, already_have_data=true\n"
                f"4. 纯闲聊? → done=true\n"
                f"5. 同方案失败2次→换思路 | Windows:dir非ls, python非python3\n"
                f"6. 禁止: 虚构完成、反复演示、创建已有工具\n"
                f"{'上轮计划: ' + plan if plan else ''}\n"
                f"{'已执行: ' + history_brief if history_brief else ''}\n"
                f"只输出一个JSON对象: {{\"done\":bool,\"plan\":\"\",\"already_have_data\":bool}}"
            )
            messages = self._messages_to_dict_list(self.memory.working_memory.messages) + [
                {"role": "user", "content": think_prompt}]
            parsed = self._call_llm_json(messages, ("done", "plan"))

            if parsed is None:
                # 解析失败（含纠错重试后）：检查熔断，否则跳出内循环走 EXECUTE
                if self.state.parse_failures >= self.MAX_PARSE_FAILURES:
                    return self._abort_to_end(
                        f"模型连续 {self.state.parse_failures} 次未按 JSON 格式输出")
                done = False
                plan_data = {}
                break
            plan_data = parsed
            done = plan_data.get("done", False)
            plan = plan_data.get("plan", "") or ""

            # 错误超限熔断：诚实中止，而非编造完成（max_iterations 由主循环断点处理）
            if self.state.error_count > self.state.max_retries:
                return self._abort_to_end(
                    f"错误已达 {self.state.error_count} 次，超过重试上限")

            # 防幻觉硬护栏：LLM 说 done 但从未真正执行工具 → 强制不通过
            if done and self.state.think_rounds < self.state.max_think_rounds:
                # 只认会产出数据/创建工具的"真执行"，info 查询不算
                real_executed = any(
                    h.get("mode") == "execute"
                    and not h.get("error")
                    and h.get("tool", "") not in self._INFO_TOOLS
                    for h in self.state.history)
                # 首轮首思即 done 且无计划 → 信任为闲聊放行
                trivially_chat = (self.state.iteration <= 1
                                  and self.state.think_rounds == 1
                                  and not plan)
                if (not real_executed and not trivially_chat
                        and self.state.guard_overrides < 2):
                    # 仅纯闲聊豁免（无实质动作词）
                    action_keywords = [
                        "创建", "写", "写入", "爬", "监控", "下载", "获取",
                        "计算", "执行", "运行", "分析", "提取", "抓取",
                        "create", "write", "fetch", "download", "monitor",
                        "analyze", "extract", "run", "execute",
                    ]
                    task_lower = self.state.task.lower()
                    needs_action = any(kw in task_lower for kw in action_keywords)
                    if needs_action:
                        done = False
                        self.state.guard_overrides += 1

            if not done:
                self.memory.add_assistant_message(
                    json.dumps(plan_data, ensure_ascii=False) if plan_data else "[think]")

        self.state.thought = plan_data.get("thought", plan)
        self.state.plan = plan

        # 程序化校验：plan 中显式提到"调用某工具"但该工具不存在 → 修正为 create_tool
        if not done and plan:
            available = set(self.tool_pool.tools.keys()) | set(self.tool_pool._live_tools.keys())
            mentioned = set(re.findall(
                r"(?:调用|使用|执行|call)\s*[`'\"]?([a-z_][a-z0-9_]{2,})", plan.lower()))
            missing = mentioned - available - {"create_tool"}
            if missing and "create_tool" not in self.state.plan.lower():
                self.state.plan = f"池中无此工具({','.join(missing)})，请调用 create_tool 创建"

        self.state.history.append({"mode": "think", "thought": self.state.thought})
        self.state.last_error = ""

        # 无进展熔断：自上次 THINK 以来无新增成功的非 INFO 工具执行
        if not done:
            progressed = False
            for h in reversed(self.state.history[:-1]):
                if h.get("mode") == "think":
                    break
                if (h.get("mode") == "execute" and not h.get("error")
                        and h.get("tool", "") not in self._INFO_TOOLS):
                    progressed = True
                    break
            if self.state.iteration > 1 and not progressed:
                self.state.no_progress_rounds += 1
            else:
                self.state.no_progress_rounds = 0
            if self.state.no_progress_rounds >= self.MAX_NO_PROGRESS:
                return self._abort_to_end(
                    f"连续 {self.state.no_progress_rounds} 轮思考无实质进展")

        if done:
            has_data = plan_data.get("already_have_data", False) if isinstance(plan_data, dict) else False
            # 如果已拿到数据但还没保存，先保存
            if has_data:
                ctx = self._messages_to_dict_list(self.memory.working_memory.messages, clean_meta=True)
                ctx.append({
                    "role": "user",
                    "content": "数据已获取，请用write_file保存到workspace，然后给出最终答案"
                })
                save_tools = [t for t in self.tool_pool.list_tools()
                              if t["function"]["name"] not in self._INFO_TOOLS
                              and t["function"]["name"] != "run_command"]
                resp = self._call_llm(ctx, save_tools, use_think_model=False)
                if resp.get("tool_calls"):
                    # 先添加 assistant+tool_calls 消息（DeepSeek 要求 tool 前必须有）
                    tcs_for_mem = [MsgToolCall(
                        id=tc.get("id", ""), type="function",
                        function=MsgFunction(name=tc["function"]["name"],
                                             arguments=tc["function"]["arguments"]))
                        for tc in resp["tool_calls"]]
                    self.memory.add_assistant_message(
                        resp.get("content") or "[tool_call]",
                        tool_calls=tcs_for_mem)
                    for tc in resp["tool_calls"]:
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            args = {}
                        r = self.tool_pool.execute(tc["function"]["name"], args)
                        print(f"  💾 {tc['function']['name']} → {r[:80]}")
                        self.memory.add_tool_message(
                            content=r, name=tc["function"]["name"],
                            tool_call_id=tc.get("id", ""))

            # 生成友好回复
            self.state.final_answer = self._generate_final_answer(self.state.task)
            return LoopMode.END

        # 未完成 → 记录思考结果，进入执行
        self.memory.add_assistant_message(plan_data.get("thought", "") or "[think]")
        return LoopMode.EXECUTE

    def _execute(self) -> LoopMode:
        """EXECUTE: 调用全部工具 → REFLECT(出错) / THINK(继续) / END(完成)"""
        exec_prompt = f"执行计划。调用工具或create_tool创建。出错说明。"
        messages = self._messages_to_dict_list(self.memory.working_memory.messages, clean_meta=True) + [
            {"role": "user", "content": exec_prompt}]

        # 双层工具列表：run_command 只在失败后作为 fallback 开放
        base_tools = [t for t in self.tool_pool.list_tools()
                      if t["function"]["name"] not in self._INFO_TOOLS]
        # 首轮 EXECUTE：隐藏 run_command，强制 LLM 用 create_tool
        if self.state.error_count == 0:
            base_tools = [t for t in base_tools
                          if t["function"]["name"] != "run_command"]

        # run_command 调用计数——超过 2 次自动屏蔽
        rc_count = sum(1 for h in self.state.history
                       if h.get("mode") == "execute"
                       and h.get("tool") == "run_command")
        if rc_count >= 2:
            base_tools = [t for t in base_tools
                          if t["function"]["name"] != "run_command"]

        response = self._call_llm(messages, base_tools, use_think_model=False)

        tool_calls = response.get("tool_calls")
        if not tool_calls:
            answer = response.get("content", "") or "完成"
            answer = self._clean_xml_artifacts(answer)
            # 防"嘴上完成"护栏：没有任何成功的实质工具执行就输出散文 → 回 THINK 复核
            real_executed = any(
                h.get("mode") == "execute" and not h.get("error")
                and h.get("tool", "") not in self._INFO_TOOLS
                for h in self.state.history)
            if not real_executed and self.state.guard_overrides < 2:
                self.state.guard_overrides += 1
                self.memory.add_assistant_message(answer)
                return LoopMode.THINK
            self.memory.add_assistant_message(answer)
            self.memory.remember_turn(self.state.task, answer)
            self.state.final_answer = answer
            self.state.result = answer
            return LoopMode.END

        # 重复调用熔断：同工具+同参数签名 ≥ MAX_REPEAT_CALLS 次 → 中止
        for tc in tool_calls:
            sig_src = tc["function"]["name"] + ":" + str(tc["function"].get("arguments", ""))
            try:
                args_norm = json.dumps(
                    json.loads(tc["function"]["arguments"]),
                    sort_keys=True, ensure_ascii=False)
                sig_src = tc["function"]["name"] + ":" + args_norm
            except (json.JSONDecodeError, TypeError):
                pass
            sig = hashlib.md5(sig_src.encode("utf-8")).hexdigest()[:12]
            count = self.state.call_signatures.get(sig, 0) + 1
            self.state.call_signatures[sig] = count
            if count >= self.MAX_REPEAT_CALLS:
                return self._abort_to_end(
                    f"检测到重复调用 {tc['function']['name']}（相同参数已 {count} 次）")

        # 检测重复无用调用：同一工具连续 2 次 → 在写入 memory 前拦截（软切换）
        recent_calls = [h for h in self.state.history[-6:]
                        if h.get("mode") == "execute"]
        this_tool = tool_calls[0]["function"]["name"] if tool_calls else ""
        if len(recent_calls) >= 2:
            last_two = recent_calls[-2:]
            if (last_two[0].get("tool") == last_two[1].get("tool") == this_tool
                    and this_tool in ("run_command", "read_file")):
                self.memory.add_assistant_message(
                    f"检测到重复{this_tool}调用，强制切换方案")
                return LoopMode.THINK

        # 构建 assistant 消息（含全部 tool_calls）
        msg_tcs = [MsgToolCall(
            id=tc.get("id", ""), type="function",
            function=MsgFunction(name=tc["function"]["name"],
                                 arguments=tc["function"]["arguments"]))
            for tc in tool_calls]
        self.memory.add_assistant_message(
            response.get("content") or "[tool_call]",
            tool_calls=msg_tcs)

        all_ok = True
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            print(f"  🔧 {func_name}({json.dumps(args, ensure_ascii=False)[:100]})")

            is_error, result = self._execute_single_tool_call(
                func_name, args, tc.get("id", ""))
            print(f"  📋 {result[:120]}{'...' if len(result) > 120 else ''}")

            self.memory.add_tool_message(content=result, name=func_name,
                                          tool_call_id=tc.get("id", ""))
            self.state.history.append(
                {"mode": "execute", "tool": func_name, "result": result,
                 "error": is_error})

            if is_error and func_name != "create_tool":
                all_ok = False
                self.state.error_count += 1
                self.state.last_error = result

            if func_name == "create_tool" and "成功" in result:
                self.state.created_tools.append({
                    "name": args.get("name", "?"),
                    "description": args.get("description", ""),
                })

        # 本轮只做了 create_tool 且成功 → 回 THINK，下一轮调用新工具获取数据
        if all(tc["function"]["name"] == "create_tool" for tc in tool_calls) and all_ok:
            return LoopMode.THINK

        if not all_ok and self.state.error_count <= self.state.max_retries:
            return LoopMode.REFLECT
        elif not all_ok:
            return LoopMode.THINK
        else:
            # 成功执行 → 直接结束（但 run_command 不算，它很少真正完成任务）
            if all_ok and not all(tc["function"]["name"] == "create_tool" for tc in tool_calls):
                if not any(tc["function"]["name"] == "run_command" for tc in tool_calls):
                    self.state.final_answer = self._generate_final_answer(self.state.task)
                    return LoopMode.END
            return LoopMode.THINK

    def _reflect(self) -> LoopMode:
        """REFLECT: 多轮反思 → EXECUTE(重试) / THINK(重新分析) / END(放弃)"""
        if self.state.error_count >= self.state.max_retries:
            self.state.final_answer = (
                f"已重试{self.state.error_count}次，无法完成。"
                f"最后错误: {self.state.last_error}")
            return LoopMode.END

        self.state.reflect_rounds = 0
        can_retry = False
        fix_plan = ""

        # "工具不存在" → 程序化强制：跳过 LLM 反思，直接创建
        if "未找到工具" in (self.state.last_error or ""):
            self.state.reflection = "工具不存在，创建它"
            self.state.history.append({"mode": "reflect", "reflection": "工具不存在→create_tool"})
            return LoopMode.EXECUTE

        while self.state.reflect_rounds < self.state.max_reflect_rounds:
            self.state.reflect_rounds += 1
            reflect_prompt = (
                f"任务执行出错({self.state.error_count}/{self.state.max_retries}次)。\n"
                f"错误: {self.state.last_error[:200]}\n"
                f"任务: {self.state.task}\n"
                f"💡 如错误是'工具不存在'→ 唯一正确方案是 create_tool 创建它\n"
                f"已有工具: {self._build_tools_brief()}\n"
                f"{'上轮反思: ' + fix_plan if fix_plan else ''}\n"
                f"反复分析直到找到可执行的解决方案。没有可行方案时can_retry=false。\n"
                f"只输出一个JSON对象: {{\"root_cause\":\"根因\",\"fix\":\"具体方案\",\"can_retry\":bool}}"
            )
            messages = self._messages_to_dict_list(
                self.memory.working_memory.messages, clean_meta=True) + [
                {"role": "user", "content": reflect_prompt}]
            fix_data = self._call_llm_json(messages, ("can_retry", "fix"))

            if fix_data is None:
                # 解析失败：检查熔断
                if self.state.parse_failures >= self.MAX_PARSE_FAILURES:
                    return self._abort_to_end(
                        f"模型连续 {self.state.parse_failures} 次未按 JSON 格式输出")
                can_retry = False
            else:
                self.memory.add_assistant_message(
                    json.dumps(fix_data, ensure_ascii=False))
                can_retry = fix_data.get("can_retry", False)
                fix_plan = fix_data.get("fix", "") or ""

            if can_retry:
                break  # 找到方案，退出反思

        self.state.reflection = fix_plan
        self.state.history.append({"mode": "reflect", "reflection": fix_plan})

        if can_retry:
            return LoopMode.EXECUTE
        return LoopMode.THINK

    # ==================== 主循环 / Turn 处理 ====================

    def process_turn(self, user_input: str, verbose: bool = True) -> str:
        """
        处理单轮用户输入（不重置记忆，适合 REPL 交互）。

        与 run() 区别：run() 调用 reset() 清空记忆，process_turn() 保留上下文。
        """
        self.memory.add_user_message(user_input)
        if verbose:
            print(f"User：{user_input}\n")

        # 初始化本 turn 状态
        self.state.task = user_input
        self.state.mode = LoopMode.THINK
        self.state.iteration = 0
        self.state.error_count = 0
        self.state.think_rounds = 0
        self.state.reflect_rounds = 0
        self.state.final_answer = ""
        self.state.result = ""
        self.state.parse_failures = 0
        self.state.call_signatures = {}
        self.state.guard_overrides = 0
        self.state.no_progress_rounds = 0

        while self.state.iteration < self.state.max_iterations:
            self.state.iteration += 1
            tool_total = self.tool_pool.get_tool_count()["total"]
            if verbose:
                print(f"📦 {tool_total} 个工具 | "
                      f"第{self.state.iteration}轮 [{self.state.mode.value}]")

            if self.state.mode == LoopMode.THINK:
                if verbose: print("🤔 思考...")
                next_mode = self._think()
            elif self.state.mode == LoopMode.EXECUTE:
                if verbose: print("⚙️ 执行...")
                next_mode = self._execute()
            elif self.state.mode == LoopMode.REFLECT:
                if verbose: print("🔍 反思...")
                next_mode = self._reflect()
            else:
                break

            self.state.mode = next_mode
            self.memory.check_and_consolidate()

            if next_mode == LoopMode.END:
                break
        else:
            self.memory.add_assistant_message(
                f"任务未完成({self.state.max_iterations}轮)")
            # 保存断点：任务 + 计划 + 已执行历史 + 错误数
            self._breakpoint = {
                "task": self.state.task,
                "plan": self.state.plan,
                "history": list(self.state.history),
                "error_count": self.state.error_count,
                "iteration": self.state.iteration,
            }
            if verbose:
                print(f"⚠️ 达到最大轮次 {self.state.max_iterations}，本轮终止。")
                print(f"💾 断点已保存，输入 /continue 或「继续」接续执行。\n")

        return self.state.final_answer or self.state.result or "任务未完成"

    def has_breakpoint(self) -> bool:
        """检查是否存在可恢复的断点"""
        return self._breakpoint is not None

    def resume(self, verbose: bool = True) -> str:
        """
        从断点恢复被终止的任务继续执行。

        复用已保存的任务、计划、执行历史，从 THINK 状态重新启动。
        保留断点的 error_count 避免无限续跑，但 iteration 重置继续计数。
        """
        if not self._breakpoint:
            msg = "没有可恢复的断点。"
            if verbose:
                print(f"❌ {msg}")
            return msg

        bp = self._breakpoint
        self._breakpoint = None  # 消费断点

        # 设置续跑提示到 memory
        self.memory.add_user_message(f"继续执行刚才未完成的任务: {bp['task']}")
        if verbose:
            print(f"🔄 续跑: {bp['task']}")
            print(f"   已执行 {len(bp.get('history', []))} 步, "
                  f"上次错误 {bp.get('error_count', 0)} 次")

        # 从断点恢复状态（保留 plan + history，重置轮数）
        self.state.task = bp["task"]
        self.state.mode = LoopMode.THINK
        self.state.iteration = 0
        self.state.error_count = bp.get("error_count", 0)
        self.state.plan = bp.get("plan", "")
        self.state.history = bp.get("history", [])
        self.state.think_rounds = 0
        self.state.reflect_rounds = 0
        self.state.final_answer = ""
        self.state.result = ""
        self.state.parse_failures = 0
        self.state.call_signatures = {}
        self.state.guard_overrides = 0
        self.state.no_progress_rounds = 0

        # 走状态机
        while self.state.iteration < self.state.max_iterations:
            self.state.iteration += 1
            tool_total = self.tool_pool.get_tool_count()["total"]
            if verbose:
                print(f"📦 {tool_total} 个工具 | "
                      f"续第{self.state.iteration}轮 [{self.state.mode.value}]")

            if self.state.mode == LoopMode.THINK:
                if verbose: print("🤔 思考...")
                next_mode = self._think()
            elif self.state.mode == LoopMode.EXECUTE:
                if verbose: print("⚙️ 执行...")
                next_mode = self._execute()
            elif self.state.mode == LoopMode.REFLECT:
                if verbose: print("🔍 反思...")
                next_mode = self._reflect()
            else:
                break

            self.state.mode = next_mode
            self.memory.check_and_consolidate()

            if next_mode == LoopMode.END:
                break
        else:
            self.memory.add_assistant_message(
                f"续跑仍未完成({self.state.max_iterations}轮)")
            # 再次保存断点，允许继续续跑
            self._breakpoint = {
                "task": self.state.task,
                "plan": self.state.plan,
                "history": list(self.state.history),
                "error_count": self.state.error_count,
                "iteration": self.state.iteration,
            }
            if verbose:
                print(f"⚠️ 续跑再次达到上限，断点已更新，可再次 /continue。\n")

        return self.state.final_answer or self.state.result or "续跑未完成"

    def run(self, task: str, verbose: bool = True) -> str:
        """一次性任务执行（兼容旧 API）。重置记忆后执行单轮。"""
        self.reset(task)

        if verbose:
            print(f"🚀 {task}")
            print(f"🧠 {self.think_model.get_model_name()} "
                  f"⚡ {self.current_model.get_model_name()}")
            print(f"🛠️ {self.tool_pool.get_tool_count()['total']} 个工具")

        result = self.process_turn(task, verbose=verbose)

        self._save_memory()

        if verbose:
            print(f"\n{'─'*35}")
            print(f"📝 {result}")

        return result

    # ==================== 持久化 ====================

    def _save_memory(self):
        try:
            self.memory.save()
        except Exception as e:
            print(f"保存记忆失败: {e}")

    def get_history(self) -> List[Dict]:
        return self.state.history

    def get_memory_count(self) -> Dict:
        return self.memory.get_stats()


# ==================== 便捷函数 ====================

def create_agent(
    model_configs: List[ModelConfig],
    system_prompt: str = "你是一个有用的AI助手，擅长分析和解决问题。",
    tools: Optional[Dict[str, Callable]] = None,
    max_iterations: int = 10,
    max_retries: int = 3,
    openai_client: Any = None,
    tool_pool: Optional[MCPToolPool] = None,
    memory: Optional[MemoryManager] = None,
) -> AgentBrain:
    """创建 AgentBrain 的便捷函数"""
    if tool_pool is None:
        tool_pool = MCPToolPool(
            pool_file="./tools/mcp_tools.json", code_dir="./tools/tool_add/tool_direct")
    if tools:
        for name, func in tools.items():
            tool_pool.add_live_tool(name, func.__doc__ or "", func)
    return AgentBrain(
        model_configs=model_configs, tool_pool=tool_pool,
        system_prompt=system_prompt, max_iterations=max_iterations,
        max_retries=max_retries, openai_client=openai_client,
        memory=memory,
    )
