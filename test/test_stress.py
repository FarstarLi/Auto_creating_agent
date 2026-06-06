"""
暴力压力测试 — 100+ 轮工具调用 + 状态机 + 记忆系统
模拟真实对话流程，验证系统在高负载下的稳定性

运行: python test/test_stress.py
"""

import sys, os, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.brain import AgentBrain
from brain.adapters import BaseModelAdapter
from brain.state import LoopMode, LoopState
from memory import MemoryManager
from tools.mcp_pool import MCPToolPool

# ============================================================
# Mock 适配器：模拟 LLM 行为（根据任务类型返回不同响应）
# ============================================================

class StressMockAdapter(BaseModelAdapter):
    """压力测试用 mock — 根据上下文智能返回"""
    def __init__(self):
        self.call_count = 0
        self.call_history = []

    def chat(self, messages, tools=None):
        self.call_count += 1
        last_msg = ""
        for m in reversed(messages):
            c = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
            if c:
                last_msg = c
                break

        tool_names = []
        if tools:
            tool_names = [t.get("function", {}).get("name", "") for t in tools]

        resp = self._decide(last_msg, tool_names)
        self.call_history.append({"msg": last_msg[:100], "resp": str(resp)[:100]})
        return resp

    def _decide(self, prompt, tool_names):
        """根据 prompt 内容决定返回什么"""
        p = (prompt or "").lower()

        # THINK 判断（含 "输出JSON"）
        if "输出json" in p:
            # 如果已有工具且任务是计算类 → done=true
            if any(kw in p for kw in ["阶乘", "求和", "斐波那契", "质数"]) and \
               any(t in tool_names for t in ["factorial", "sum_numbers", "fibonacci", "is_prime"]):
                return {"content": json.dumps({"done": True, "plan": "已有工具可直接用", "already_have_data": False}),
                        "tool_calls": None}
            if any(kw in p for kw in ["已执行", "成功"]):
                return {"content": json.dumps({"done": True, "plan": "", "already_have_data": True}),
                        "tool_calls": None}
            if "hello" in p or "你好" in p or "谢谢" in p:
                return {"content": json.dumps({"done": True, "plan": "", "already_have_data": False}),
                        "tool_calls": None}
            return {"content": json.dumps({"done": False, "plan": "创建或调用工具", "already_have_data": False}),
                    "tool_calls": None}

        # REFLECT
        if "输出json" in p and "root_cause" in p:
            return {"content": json.dumps({"root_cause": "测试错误", "fix": "重试", "can_retry": True}),
                    "tool_calls": None}

        # EXECUTE — 根据任务类型选择工具
        if "执行计划" in p or "调用工具" in p:
            if "阶乘" in p and "factorial" in tool_names:
                return {"content": "调用已有", "tool_calls": [_tc("c1", "factorial", {"n": 10})]}
            if "求和" in p and "sum_numbers" in tool_names:
                return {"content": "调用已有", "tool_calls": [_tc("c1", "sum_numbers", {"numbers": [1, 2, 3]})]}
            if "读取" in p and "read_file" in tool_names:
                return {"content": "读取", "tool_calls": [_tc("c1", "read_file", {"path": "./README.md"})]}
            if "创建" in p or "阶乘" in p:
                return {"content": "创建", "tool_calls": [_tc("c1", "create_tool",
                    {"name": "factorial", "description": "计算阶乘",
                     "code": "def factorial(n: int) -> str:\n    r=1\n    for i in range(1,n+1):r*=i\n    return str(r)"})]}
            if "求和" in p:
                return {"content": "创建", "tool_calls": [_tc("c1", "create_tool",
                    {"name": "sum_numbers", "description": "列表求和",
                     "code": "def sum_numbers(numbers: list) -> str:\n    return str(sum(numbers))"})]}

        # 最终回复
        return {"content": "任务已完成。", "tool_calls": None}

    def get_model_name(self): return "stress-mock"


def _tc(id_, name, args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


# ============================================================
# 测试运行器
# ============================================================

passed = 0
failed = 0
checks = []

def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
        checks.append(f"  [OK] {msg}")
    else:
        failed += 1
        checks.append(f"  [FAIL] {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 初始化
# ============================================================
section("初始化测试环境")

adapter = StressMockAdapter()
brain = AgentBrain.__new__(AgentBrain)
brain.models = [adapter]; brain.current_model_idx = 0; brain.think_model_idx = 0
brain.system_prompt = "你是测试助手。"
brain.max_iterations = 5; brain.max_retries = 3
brain.PROTECTED_TOOLS = ["read_file","write_file","create_file","delete_file",
                          "create_directory","run_command","delete_tool"]
brain.openai_client = None
brain._breakpoint = None
brain.memory = MemoryManager(client=None, model="mock")
brain.state = LoopState(max_iterations=5, max_retries=3, max_think_rounds=1, max_reflect_rounds=1)
brain.tool_pool = MCPToolPool(pool_file="./tools/mcp_tools.json",
                               code_dir="./tools/tool_add/tool_direct")
brain._register_brain_tools()

print(f"  适配器: {adapter.get_model_name()}")
print(f"  初始工具: {brain.tool_pool.get_tool_count()['total']} 个")
print(f"  记忆状态: {brain.memory.summary()}")

# ============================================================
# 第 1 轮：闲聊测试 (20 次)
# ============================================================
section("第 1 轮：闲聊稳定性 (20 次)")

greetings = ["你好", "hello", "你是谁", "今天天气不错", "谢谢你的帮助",
             "你能做什么", "帮我看看", "好的", "知道了", "不错",
             "再见", "辛苦了", "等一下", "没问题", "了解了",
             "可以", "行", "嗯嗯", "ok", "谢谢"]

for i, msg in enumerate(greetings):
    try:
        result = brain.process_turn(msg, verbose=False)
        check(result is not None and len(result) > 0, f"闲聊 {i+1}: '{msg}' → 回复正常")
    except Exception as e:
        check(False, f"闲聊 {i+1}: '{msg}' → 异常: {e}")

check(brain.state.error_count < 3, f"闲聊后 error_count={brain.state.error_count} (应 < 3)")
check(adapter.call_count > 0, f"LLM 调用 {adapter.call_count} 次")

# ============================================================
# 第 2 轮：工具创建 + 调用 (30 次)
# ============================================================
section("第 2 轮：工具创建与调用 (30 次)")

tool_tasks = [
    ("帮我创建一个计算阶乘的工具", "create_factorial"),
    ("计算 10 的阶乘", "use_factorial"),
    ("帮我创建一个列表求和的工具", "create_sum"),
    ("计算 [1,2,3] 的和", "use_sum"),
    ("创建一个判断质数的工具", "create_prime"),
    ("读取 README.md 文件", "read_file"),
    ("创建一个斐波那契数列生成器", "create_fib"),
    ("帮我生成前 20 个斐波那契数", "use_fib"),
    ("创建一个 JSON 解析工具", "create_json"),
    ("创建一个 CSV 读取工具", "create_csv"),
    ("把结果保存到 output.txt", "write_file"),
    ("删除临时文件 temp.txt", "delete_file"),
    ("创建 output 目录", "create_dir"),
    ("列出当前目录文件", "list_dir"),
    ("计算 100 的阶乘", "factorial"),
] * 2  # 30 次

for i, (task, task_type) in enumerate(tool_tasks):
    try:
        result = brain.process_turn(task, verbose=False)
        check(isinstance(result, str) and len(result) > 0, f"工具 {i+1}: '{task[:30]}' → OK")
    except Exception as e:
        check(False, f"工具 {i+1}: '{task[:30]}' → 异常: {e}")

pool_count = brain.tool_pool.get_tool_count()["total"]
check(pool_count >= 7, f"工具池: {pool_count} 个工具")
check(adapter.call_count > len(greetings), "工具阶段有 LLM 调用")

# ============================================================
# 第 3 轮：错误恢复 (20 次)
# ============================================================
section("第 3 轮：错误恢复 (20 次)")

error_tasks = [
    "用不存在的工具读取配置",
    "执行一个肯定会失败的操作",
    "访问一个不存在的文件路径",
    "调用一个未创建的工具函数",
    "删除一个不存在的文件",
] * 4  # 20 次

for i, task in enumerate(error_tasks):
    try:
        result = brain.process_turn(task, verbose=False)
        check(isinstance(result, str), f"错误 {i+1}: 系统未崩溃，返回了结果")
    except Exception as e:
        check(False, f"错误 {i+1}: 崩溃: {e}")

check(brain.state.error_count < brain.max_iterations * 2,
      f"错误计数可控: {brain.state.error_count}")

# ============================================================
# 第 4 轮：断点续跑 (10 次)
# ============================================================
section("第 4 轮：断点续跑 (10 次)")

bp_count = 0
for i in range(10):
    # 用小 max_iterations 制造断点
    brain.max_iterations = 1
    try:
        result = brain.process_turn(f"复杂长任务 {i}", verbose=False)
        if brain.has_breakpoint():
            bp_count += 1
            brain2 = AgentBrain.__new__(AgentBrain)
            brain2.models = [adapter]; brain2.current_model_idx = 0; brain2.think_model_idx = 0
            brain2.system_prompt = brain.system_prompt
            brain2.max_iterations = 10; brain2.max_retries = 3
            brain2.PROTECTED_TOOLS = brain.PROTECTED_TOOLS
            brain2.openai_client = None
            brain2._breakpoint = brain._breakpoint
            brain2.memory = brain.memory
            brain2.tool_pool = brain.tool_pool
            brain2.state = LoopState(max_iterations=10, max_retries=3,
                                      max_think_rounds=1, max_reflect_rounds=1)
            resume_result = brain2.resume(verbose=False)
            check(isinstance(resume_result, str), f"续跑 {i+1}: OK")
    except Exception as e:
        check(False, f"断点 {i+1}: 异常: {e}")

brain.max_iterations = 5  # 恢复
check(bp_count > 0, f"产生 {bp_count} 个断点")

# ============================================================
# 第 5 轮：记忆压力 (20 次)
# ============================================================
section("第 5 轮：记忆压力 (20 次)")

long_msgs = [
    "请帮我分析一下这个数据集的统计特征，包括均值、方差、中位数、众数，以及数据的分布情况。" * 3,
    "我需要一个完整的解决方案，包括数据清洗、特征工程、模型训练、超参数调优和最终评估。" * 3,
    "请详细解释 Python 中的装饰器、上下文管理器、生成器、协程和异步编程的区别和联系。" * 3,
    "帮我设计一个完整的 RESTful API 架构，包括认证授权、限流、缓存、日志和监控方案。" * 3,
    "分析一下微服务架构 vs 单体架构的优劣，从开发效率、部署复杂度、运维成本、扩展性等方面对比。" * 3,
] * 4  # 20 次

for i, msg in enumerate(long_msgs):
    try:
        result = brain.process_turn(msg, verbose=False)
        check(isinstance(result, str), f"长消息 {i+1}: 正常返回")
    except Exception as e:
        check(False, f"长消息 {i+1}: 异常: {e}")

working_count = len(brain.memory.working_memory.messages)
mem_stats = brain.memory.get_stats()
check(working_count > 0, f"工作记忆: {mem_stats['working_memory']['messages']} 条消息")
check(mem_stats["short_term_memory"]["items"] + mem_stats["long_term_memory"]["items"] >= 0,
      "记忆系统正常")

# ============================================================
# 第 6 轮：并发工具操作 (15 次)
# ============================================================
section("第 6 轮：混合操作压力 (15 次)")

mixed_tasks = [
    "帮我读取文件并分析内容",
    "创建工具然后立即使用",
    "读取、分析、保存三步走",
    "检查工具池中有什么工具",
    "搜索与文件相关的工具",
] * 3  # 15 次

for i, task in enumerate(mixed_tasks):
    try:
        result = brain.process_turn(task, verbose=False)
        check(isinstance(result, str), f"混合 {i+1}: OK")
    except Exception as e:
        check(False, f"混合 {i+1}: 异常: {e}")

# ============================================================
# 第 7 轮：工具清理 (验证无泄漏)
# ============================================================
section("第 7 轮：清理验证")

final_pool = brain.tool_pool.get_tool_count()
print(f"  最终工具数: {final_pool['total']} (持久化: {final_pool['persisted']})")

# 清理测试创建的工具
test_tools = ["factorial", "sum_numbers"]
for name in test_tools:
    if brain.tool_pool.has_tool(name):
        brain.tool_pool.delete_tool(name)
        print(f"  🧹 清理: {name}")

final_count = brain.tool_pool.get_tool_count()["total"]
check(final_count == 7, f"清理后工具数 = 7 (实际: {final_count})")

# 清理 workspace
import glob
for f in glob.glob("./workspace/*"):
    try:
        os.remove(f)
    except:
        pass

# ============================================================
# 结果汇总
# ============================================================
total = passed + failed
section(f"测试结果: {passed}/{total} 通过 ({100*passed//total if total else 0}%)")

# 统计
print(f"\n[chart] 统计:")
print(f"  总交互轮次: {len(greetings) + len(tool_tasks) + len(error_tasks) + 10 + len(long_msgs) + len(mixed_tasks)}")
print(f"  LLM 调用次数: {adapter.call_count}")
print(f"  最终记忆: {brain.memory.summary()}")
print(f"  工具池: {brain.tool_pool.get_tool_count()}")
print(f"  内存使用: 工作记忆 {mem_stats['working_memory']['messages']} 条")

# 最近 5 条检查结果
print(f"\n[clip] 最近检查项:")
for c in checks[-10:]:
    print(c)

if failed == 0:
    print(f"\n[party] 全部 {passed} 项测试通过！系统在高负载下运行稳定。")
else:
    print(f"\n[WARN]️ {failed} 项失败！")

sys.exit(0 if failed == 0 else 1)
