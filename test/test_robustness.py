"""
防幻觉/防死循环加固测试 —— json_utils 解析 + 状态机熔断 + 回归

覆盖:
  - json_utils: 围栏剥离 / 平衡括号扫描 / 期望键校验
  - THINK 解析失败纠错重试
  - 解析熔断 / 重复调用签名熔断
  - 错误前缀判定（替代子串误判）
  - create_tool 后回 THINK（不再虚假 END）
  - _inject_rules 查重生效
  - 闲聊护栏放行

运行方式:
    cd c:/Users/qq215/Desktop/auto_coding
    python -m pytest test/test_robustness.py -v
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from json_utils import extract_json, parse_llm_json
from brain.brain import AgentBrain
from brain.adapters import BaseModelAdapter
from brain.state import LoopMode, LoopState
from memory import MemoryManager


# ============================================================
# Mock Adapter（带调用上限保险丝，防测试自身死循环）
# ============================================================

class FuseMockAdapter(BaseModelAdapter):
    """返回预定义响应；超过 max_calls 次调用直接断言失败"""

    def __init__(self, responses=None, max_calls=40, repeat_last=False):
        self.responses = responses or []
        self.call_idx = 0
        self.call_history = []
        self.max_calls = max_calls
        self.repeat_last = repeat_last  # 耗尽后重复最后一个响应（模拟顽固模型）

    def chat(self, messages, tools=None, json_mode=False):
        assert len(self.call_history) < self.max_calls, (
            f"MockAdapter 调用超过 {self.max_calls} 次——疑似死循环！")
        self.call_history.append({
            "messages": messages, "tools": tools, "json_mode": json_mode})
        if self.call_idx < len(self.responses):
            resp = self.responses[self.call_idx]
            self.call_idx += 1
            return resp
        if self.repeat_last and self.responses:
            return self.responses[-1]
        return {"content": json.dumps(
            {"done": True, "plan": "", "already_have_data": False}),
            "tool_calls": None}

    def get_model_name(self):
        return "fuse-mock"


def PROSE(text="好的，我认为这个任务需要先分析一下，然后再决定怎么做。"):
    """快捷：非 JSON 散文响应（模拟幻觉输出）"""
    return {"content": text, "tool_calls": None}


def NONE_CONTENT():
    """快捷：content 为 None（曾导致 json.loads(None) 崩溃）"""
    return {"content": None, "tool_calls": None}


def DONE(already_have_data=False):
    return {"content": json.dumps(
        {"done": True, "plan": "", "already_have_data": already_have_data}),
        "tool_calls": None}


def FENCED_DONE():
    """快捷：markdown 围栏包裹的 done JSON（常见幻觉形态）"""
    inner = json.dumps({"done": True, "plan": "", "already_have_data": False})
    return {"content": f"```json\n{inner}\n```", "tool_calls": None}


def NOT_DONE(plan="需要工具"):
    return {"content": json.dumps(
        {"done": False, "plan": plan, "already_have_data": False}),
        "tool_calls": None}


def TOOL_CALL(id_, name, args):
    return {"content": "调用工具",
            "tool_calls": [{"id": id_, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def FRIENDLY(text="任务完成！"):
    return {"content": text, "tool_calls": None}


def make_brain(responses=None, max_iterations=10, max_retries=3,
               max_think_rounds=1, max_reflect_rounds=1,
               max_calls=40, repeat_last=False):
    """创建带保险丝 mock 适配器的 AgentBrain"""
    adapter = FuseMockAdapter(responses, max_calls=max_calls, repeat_last=repeat_last)
    brain = AgentBrain.__new__(AgentBrain)
    brain.models = [adapter]
    brain.current_model_idx = 0
    brain.think_model_idx = 0
    brain.system_prompt = "你是一个测试助手。"
    brain.max_iterations = max_iterations
    brain.max_retries = max_retries
    brain.PROTECTED_TOOLS = [
        "read_file", "write_file", "create_file", "delete_file",
        "create_directory", "run_command", "delete_tool",
    ]
    brain.openai_client = None
    brain._breakpoint = None
    brain.memory = MemoryManager(client=None, model="mock")
    brain.state = LoopState(max_iterations=max_iterations, max_retries=max_retries,
                            max_think_rounds=max_think_rounds,
                            max_reflect_rounds=max_reflect_rounds)

    from tools.mcp_pool import MCPToolPool
    brain.tool_pool = MCPToolPool(
        pool_file="./tools/mcp_tools.json",
        code_dir="./tools/tool_add/tool_direct",
    )
    return brain, adapter


# ============================================================
# 1. json_utils 单测
# ============================================================

class TestExtractJson:
    """extract_json 健壮性"""

    def test_bare_json(self):
        assert extract_json('{"done": true}') == {"done": True}

    def test_fenced_json(self):
        text = '```json\n{"done": true, "plan": "x"}\n```'
        assert extract_json(text) == {"done": True, "plan": "x"}

    def test_fenced_no_lang_tag(self):
        text = '```\n{"a": 1}\n```'
        assert extract_json(text) == {"a": 1}

    def test_prose_wrapped_json(self):
        text = '好的，我的计划是 {"done": false, "plan": "创建工具"} 希望对你有帮助！'
        assert extract_json(text) == {"done": False, "plan": "创建工具"}

    def test_braces_inside_string_value(self):
        text = '{"plan": "写一个 {name: value} 格式的解析器", "done": false}'
        data = extract_json(text)
        assert data["plan"] == "写一个 {name: value} 格式的解析器"

    def test_escaped_quotes_inside_string(self):
        text = '前言 {"plan": "他说\\"不行\\"然后退出", "done": true} 后记'
        data = extract_json(text)
        assert data["done"] is True

    def test_multiple_json_takes_first(self):
        text = '{"a": 1} 然后 {"b": 2}'
        assert extract_json(text) == {"a": 1}

    def test_pure_prose_returns_none(self):
        assert extract_json("我觉得这个任务很简单，不需要工具。") is None

    def test_none_and_empty(self):
        assert extract_json(None) is None
        assert extract_json("") is None
        assert extract_json("   ") is None

    def test_non_string_input(self):
        assert extract_json(123) is None
        assert extract_json({"already": "dict"}) is None

    def test_json_array_rejected(self):
        # 只接受 dict（状态机期望对象）
        assert extract_json('[1, 2, 3]') is None

    def test_incomplete_json(self):
        assert extract_json('{"done": true, "plan":') is None


class TestParseLlmJson:
    """parse_llm_json 期望键校验"""

    def test_expected_key_hit(self):
        assert parse_llm_json('{"done": true}', ("done", "plan")) == {"done": True}

    def test_expected_key_miss(self):
        # 提取到无关 JSON（不含期望键）→ None
        assert parse_llm_json('{"foo": 1}', ("done", "plan")) is None

    def test_no_expected_keys_passes(self):
        assert parse_llm_json('{"foo": 1}') == {"foo": 1}


# ============================================================
# 2-4. 解析失败重试 / 熔断 / 围栏走通
# ============================================================

class TestParseRetry:
    """THINK 解析失败 → 纠错重试"""

    def test_prose_then_valid_json_recovers(self):
        """第1次散文 → 纠错重试拿到合法 JSON → 正常完成"""
        brain, adapter = make_brain([
            PROSE("我想想看..."),     # THINK 第1次：散文
            DONE(),                   # 纠错重试：合法 JSON
            FRIENDLY("你好！"),       # 最终回复
        ])
        result = brain.process_turn("你好", verbose=False)
        assert "你好" in result
        assert not brain.has_breakpoint()
        # 验证发生了纠错重试：第2次调用的消息中含纠错指令
        retry_msgs = adapter.call_history[1]["messages"]
        assert any("不是合法" in str(m.get("content", "")) for m in retry_msgs)

    def test_none_content_does_not_crash(self):
        """content=None 不再抛 TypeError 崩溃"""
        brain, _ = make_brain([
            NONE_CONTENT(),
            DONE(),
            FRIENDLY("没事了"),
        ])
        result = brain.process_turn("你好", verbose=False)
        assert result  # 正常返回，没有异常

    def test_fenced_json_works(self):
        """markdown 围栏包裹的 JSON 正常走通"""
        brain, _ = make_brain([
            FENCED_DONE(),
            FRIENDLY("围栏也能解析！"),
        ])
        result = brain.process_turn("你好", verbose=False)
        assert "围栏" in result


class TestParseFuse:
    """解析熔断：连续失败 → 中止 + 断点，绝不死循环"""

    def test_persistent_prose_aborts_with_breakpoint(self):
        """模型顽固输出散文 → 有限次调用内中止并保存断点"""
        brain, adapter = make_brain(
            [PROSE("我拒绝输出 JSON。")],
            repeat_last=True, max_calls=30, max_iterations=10)
        result = brain.process_turn("帮我创建一个爬虫工具", verbose=False)
        assert "中止" in result or "未完成" in result
        assert brain.has_breakpoint(), "熔断后应保存断点"
        # 关键：远低于保险丝上限（不是靠 max_iterations 苟活）
        assert len(adapter.call_history) < 30


class TestRepeatCallFuse:
    """重复调用签名熔断"""

    def test_same_tool_same_args_aborts(self):
        """同工具+同参数（失败路径反复重试）≥3 次 → 中止 + 断点

        成功的调用会正常 END，重复只发生在失败循环：
        EXECUTE(失败) → REFLECT(工具不存在→程序化直回 EXECUTE) → 同样的调用...
        """
        same_call = TOOL_CALL("c1", "no_such_tool_xyz", {"q": "1"})
        brain, adapter = make_brain(
            max_think_rounds=1, max_iterations=20, max_retries=10,
            max_calls=30, repeat_last=True,
            responses=[
                NOT_DONE("调用工具"),
                same_call,   # 第1次：未找到工具 → REFLECT 程序化直回 EXECUTE
                same_call,   # 第2次：同上
                same_call,   # 第3次：签名熔断 → abort
            ])
        result = brain.process_turn("反复调用不存在的工具", verbose=False)
        assert "重复调用" in result
        assert brain.has_breakpoint()
        assert brain._breakpoint.get("abort_reason", "").startswith("检测到重复调用")


# ============================================================
# 6. 错误前缀判定
# ============================================================

class TestErrorPrefix:
    """前缀约定替代子串误判"""

    def test_file_content_with_error_word_not_misjudged(self):
        """文件内容含'错误'二字 → 不算执行失败"""
        content = "本文讨论常见的代码错误类型，包括语法错误和逻辑错误。"
        assert AgentBrain._is_error_result(content) is False

    def test_real_error_prefix_detected(self):
        assert AgentBrain._is_error_result("错误：未找到工具 'xxx'") is True
        assert AgentBrain._is_error_result("失败：无法连接") is True
        assert AgentBrain._is_error_result("🚫 安全限制：命令被禁止") is True
        assert AgentBrain._is_error_result("安全警告：禁止导入 os") is True
        assert AgentBrain._is_error_result("读取文件失败：not found") is True
        assert AgentBrain._is_error_result("错误：执行工具 'x' 失败: boom") is True

    def test_success_results_pass(self):
        assert AgentBrain._is_error_result("文件创建成功：a.txt") is False
        assert AgentBrain._is_error_result("✅ 工具 'x' 创建成功！") is False

    def test_leading_whitespace_handled(self):
        assert AgentBrain._is_error_result("  错误：x") is True


# ============================================================
# 7-9. 回归测试
# ============================================================

class TestCreateToolFlow:
    """create_tool 成功后回 THINK，不再虚假 END"""

    def test_create_tool_returns_to_think(self):
        code = "def robust_tmp_tool() -> str:\n    return 'ok'\n"
        brain, _ = make_brain(
            max_think_rounds=1, max_iterations=10,
            responses=[
                NOT_DONE("创建工具"),
                TOOL_CALL("c1", "create_tool", {
                    "name": "robust_tmp_tool", "description": "临时测试",
                    "code": code}),
                NOT_DONE("调用新工具"),                      # 回 THINK（而非直接 END）
                TOOL_CALL("c2", "robust_tmp_tool", {}),      # 真正调用获取数据
                DONE(),
                FRIENDLY("工具已创建并调用"),
            ])
        try:
            result = brain.process_turn("创建一个临时工具", verbose=False)
            modes = [(h.get("mode"), h.get("tool")) for h in brain.state.history]
            # create_tool 之后存在 think，且新工具被真正调用过
            tools_called = [t for m, t in modes if m == "execute"]
            assert "create_tool" in tools_called
            assert "robust_tmp_tool" in tools_called, \
                f"新工具应被调用: {modes}"
        finally:
            brain.tool_pool.delete_tool("robust_tmp_tool")


class TestInjectRules:
    """_inject_rules 查重生效"""

    def test_rules_injected_once(self):
        brain, _ = make_brain()
        long_msg = [{"role": "user", "content": "x" * 3000}]
        once = brain._inject_rules(long_msg)
        twice = brain._inject_rules(once)
        rule_count = sum(
            1 for m in twice
            if AgentBrain._RULES_MARKER in str(m.get("content", "")))
        assert rule_count == 1, f"规则块应只注入一次，实际 {rule_count} 个"

    def test_short_messages_not_injected(self):
        brain, _ = make_brain()
        short = [{"role": "user", "content": "你好"}]
        assert brain._inject_rules(short) == short


class TestChitchatGuard:
    """闲聊护栏放行"""

    def test_chitchat_with_action_word_not_stuck(self):
        """闲聊问题含'计算'二字 → 首轮 done + 空 plan 应放行，不被护栏卡死"""
        brain, adapter = make_brain([
            DONE(),                       # 首轮首思即 done，plan 为空 → 信任放行
            FRIENDLY("这是个哲学问题～"),
        ])
        result = brain.process_turn("你觉得计算机有意识吗", verbose=False)
        assert "哲学" in result
        assert not brain.has_breakpoint()
        assert len(adapter.call_history) == 2  # 没有被护栏拖入额外循环


# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
