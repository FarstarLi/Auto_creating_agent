"""
AgentBrain 四状态机完整测试（Mock LLM）
覆盖 THINK→EXECUTE⇄REFLECT→END + 断点续跑

运行方式:
    cd c:/Users/qq215/Desktop/auto_coding
    python -m pytest test/test_state_machine.py -v
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain.brain import AgentBrain
from brain.adapters import ModelConfig, ModelProvider, BaseModelAdapter
from brain.state import LoopMode, LoopState
from memory import MemoryManager
from memory.models import Message, ToolCall, Function


# ============================================================
# Mock Adapter
# ============================================================

class MockAdapter(BaseModelAdapter):
    """返回预定义响应的适配器"""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_idx = 0
        self.call_history = []

    def chat(self, messages, tools=None):
        self.call_history.append({"tools": tools})
        if self.call_idx < len(self.responses):
            resp = self.responses[self.call_idx]
            self.call_idx += 1
            return resp
        return {"content": json.dumps({"done": True, "plan": "", "already_have_data": False}),
                "tool_calls": None}

    def get_model_name(self):
        return "mock"


def DONE(already_have_data=False):
    """快捷：返回 done=True"""
    return {"content": json.dumps({"done": True, "plan": "", "already_have_data": already_have_data}),
            "tool_calls": None}

def NOT_DONE(plan="需要工具"):
    """快捷：返回 done=False"""
    return {"content": json.dumps({"done": False, "plan": plan, "already_have_data": False}),
            "tool_calls": None}

def CAN_RETRY(fix="重试方案"):
    """快捷：返回 can_retry=True"""
    return {"content": json.dumps({"root_cause": "错误", "fix": fix, "can_retry": True}),
            "tool_calls": None}

def CANNOT_RETRY():
    """快捷：返回 can_retry=False"""
    return {"content": json.dumps({"root_cause": "无解", "fix": "", "can_retry": False}),
            "tool_calls": None}

def TOOL_CALL(id_, name, args):
    """快捷：返回带 tool_call 的响应"""
    return {"content": "调用工具",
            "tool_calls": [{"id": id_, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}

def FRIENDLY(text="任务完成！"):
    """快捷：友好回复"""
    return {"content": text, "tool_calls": None}


# ============================================================
# 工厂函数
# ============================================================

def make_brain(responses=None, max_iterations=10, max_retries=3,
               max_think_rounds=1, max_reflect_rounds=1):
    """创建带 mock 适配器的 AgentBrain"""
    adapter = MockAdapter(responses)
    brain = AgentBrain.__new__(AgentBrain)
    brain.models = [adapter]
    brain.current_model_idx = 0
    brain.think_model_idx = 0
    brain.tool_pool = None
    brain.tools = None
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
# 核心流程测试
# ============================================================

class TestHappyPath:
    """正常流程测试"""

    def test_simple_greeting(self):
        """闲聊：THINK 直接 done → END"""
        brain, _ = make_brain([DONE(), FRIENDLY("你好！有什么可以帮你的？")])
        result = brain.process_turn("你好", verbose=False)
        assert "你好" in result

    def test_tool_task_completes(self):
        """工具任务：THINK→EXECUTE→THINK→END"""
        brain, _ = make_brain([
            NOT_DONE("读取文件"),
            TOOL_CALL("c1", "read_file", {"path": "./README.md"}),
            DONE(already_have_data=True),
            FRIENDLY("文件已读取！"),
        ])
        result = brain.process_turn("读取 README.md", verbose=False)
        assert result != "任务未完成"
        modes = [h["mode"] for h in brain.state.history]
        assert "execute" in modes, f"应包含 execute: {modes}"
        assert "think" in modes, f"应包含 think: {modes}"

    def test_no_tool_needed_ends_directly(self):
        """LLM 直接回答（无需工具）→ END"""
        brain, _ = make_brain([
            NOT_DONE("直接回答"),
            FRIENDLY("答案是42"),
        ])
        result = brain.process_turn("1+1=?", verbose=False)
        assert result == "答案是42"


class TestErrorRecovery:
    """错误恢复流程测试"""

    def test_execute_error_reflect_retry(self):
        """EXECUTE 出错 → REFLECT 找方案 → EXECUTE 重试 → 成功"""
        brain, _ = make_brain(max_reflect_rounds=2, responses=[
            NOT_DONE("调用工具"),
            TOOL_CALL("c1", "nonexistent_tool", {"x": "1"}),
            CAN_RETRY("换用 read_file"),
            TOOL_CALL("c2", "read_file", {"path": "./README.md"}),
            DONE(already_have_data=True),
            FRIENDLY("重试成功了！"),
        ])
        result = brain.process_turn("测试错误恢复", verbose=False)
        assert result != "任务未完成"
        modes = [h["mode"] for h in brain.state.history]
        assert "reflect" in modes, f"应包含 reflect: {modes}"

    def test_reflect_cannot_retry_goes_to_think(self):
        """REFLECT 找不到方案 → 回到 THINK 重新规划"""
        brain, _ = make_brain([
            NOT_DONE("调用工具"),
            TOOL_CALL("c1", "nonexistent_tool", {"x": "1"}),
            CANNOT_RETRY(),
            DONE(),
            FRIENDLY("无法完成，给你替代方案"),
        ])
        result = brain.process_turn("不可能的任务", verbose=False)
        assert result != "任务未完成"
        modes = [h["mode"] for h in brain.state.history]
        assert "reflect" in modes

    def test_multi_round_reflect_finds_fix_eventually(self):
        """REFLECT 第1轮无方案 → 第2轮找到 → 重试"""
        brain, _ = make_brain(max_reflect_rounds=2, responses=[
            NOT_DONE("试工具"),
            TOOL_CALL("c1", "bad_tool", {}),
            CANNOT_RETRY(),  # 第1轮反思：无方案
            NOT_DONE("再试别的"),  # THINK：重新规划
            TOOL_CALL("c2", "bad_tool2", {}),
            CAN_RETRY("用 read_file"),  # 第2次 REFLECT：有方案
            TOOL_CALL("c3", "read_file", {"path": "./README.md"}),
            DONE(already_have_data=True),
            FRIENDLY("终于成功了"),
        ])
        result = brain.process_turn("困难任务", verbose=False)
        assert result != "任务未完成"


class TestBreakpoint:
    """断点续跑测试"""

    def test_max_iterations_saves_breakpoint(self):
        """达到 max_iterations → 保存断点"""
        # 用 2 轮限制，2 个响应 → 第1轮 THINK(not done) 第2轮 EXECUTE(success)
        # 循环结束 → else 子句保存断点
        # max_iterations=1 → THINK(NOT_DONE)→EXECUTE→循环退出→保存断点
        brain, _ = make_brain(max_iterations=1, responses=[
            NOT_DONE("需要工具"),
        ])
        brain.process_turn("超限测试", verbose=False)
        assert brain.has_breakpoint(), "超限后应有断点"
        assert brain._breakpoint["task"] == "超限测试"
        assert len(brain._breakpoint["history"]) > 0

    def test_resume_from_breakpoint_completes(self):
        """从断点恢复并完成"""
        brain1, _ = make_brain(max_iterations=1, responses=[
            NOT_DONE("需要工具"),
        ])
        brain1.process_turn("长任务", verbose=False)
        assert brain1.has_breakpoint()

        # 用新 brain 续跑
        brain2, _ = make_brain(max_iterations=10, responses=[
            DONE(),
            FRIENDLY("续跑完成！"),
        ])
        brain2._breakpoint = brain1._breakpoint
        brain2.memory = brain1.memory

        result = brain2.resume(verbose=False)
        assert result != "任务未完成"
        assert not brain2.has_breakpoint()

    def test_error_count_preserved(self):
        """续跑保留 error_count"""
        brain, _ = make_brain(max_iterations=1, responses=[NOT_DONE("x")])
        brain.state.error_count = 3
        brain.state.history = [{"mode": "execute", "tool": "bad", "error": True}]
        brain._breakpoint = {
            "task": "t", "plan": "p", "history": brain.state.history,
            "error_count": 3, "iteration": 1,
        }

        brain2, _ = make_brain(responses=[DONE(), FRIENDLY("ok")])
        brain2._breakpoint = brain._breakpoint
        brain2.memory = brain.memory
        brain2.resume(verbose=False)
        assert brain2.state.error_count == 3

    def test_no_breakpoint_returns_message(self):
        """无断点时 resume 返回提示"""
        brain, _ = make_brain()
        assert "没有可恢复的断点" in brain.resume(verbose=False)

    def test_resume_consumes_breakpoint(self):
        """resume 后断点被消费"""
        brain, _ = make_brain(max_iterations=1, responses=[NOT_DONE("x")])
        brain.process_turn("test", verbose=False)
        assert brain.has_breakpoint()

        brain2, _ = make_brain(responses=[DONE(), FRIENDLY("ok")])
        brain2._breakpoint = brain._breakpoint
        brain2.memory = brain.memory
        brain2.resume(verbose=False)
        assert not brain2.has_breakpoint()


class TestMultiRoundThink:
    """多轮思考测试"""

    def test_think_two_rounds_then_done(self):
        """THINK 第1轮不确定 → 第2轮确认 → END"""
        brain, _ = make_brain(max_think_rounds=2, responses=[
            NOT_DONE("再想想"),
            DONE(),
            FRIENDLY("想清楚了"),
        ])
        result = brain.process_turn("复杂问题", verbose=False)
        assert result != "任务未完成"

    def test_already_have_data_triggers_save(self):
        """has_data=True → 先保存再回复"""
        brain, _ = make_brain([
            NOT_DONE("获取数据"),
            TOOL_CALL("c1", "read_file", {"path": "./README.md"}),
            DONE(already_have_data=True),
            FRIENDLY("数据已保存"),
        ])
        result = brain.process_turn("获取并保存数据", verbose=False)
        assert result != "任务未完成"


class TestToolAutoCleanup:
    """工具自动清理测试"""

    def test_create_tool_conflict_auto_delete(self):
        """create_tool 重名时自动删旧"""
        brain, _ = make_brain()
        pool = brain.tool_pool
        code = "def test_cleanup_tool() -> str:\n    return 'ok'\n"
        pool.register_tool("test_cleanup_tool", "测试", code)
        r = pool.register_tool("test_cleanup_tool", "测试v2", code)
        assert "已存在" in r or "成功" in r
        pool.delete_tool("test_cleanup_tool")

    def test_failed_tool_not_auto_deleted(self):
        """失败的工具不会被自动删除——LLM 应主动调用 delete_tool"""
        brain, _ = make_brain()
        pool = brain.tool_pool
        r = pool.register_tool("will_fail_test", "会失败",
            "def will_fail_test() -> str:\n    raise Exception('x')\n")
        if "成功" in r:
            is_error, _ = brain._execute_single_tool_call("will_fail_test", {}, "")
            assert is_error
            # 工具仍然存在——不再自动删除
            assert pool.has_tool("will_fail_test")
            pool.delete_tool("will_fail_test")

    def test_protected_tools_always_exist(self):
        """内置工具始终存在"""
        brain, _ = make_brain()
        assert "read_file" in brain.PROTECTED_TOOLS
        assert brain.tool_pool.has_tool("read_file")


class TestCleanXmlArtifacts:
    """XML 清理测试"""

    def test_nested_xml_cleaned(self):
        brain, _ = make_brain()
        dirty = "你好<function_calls><invoke name=\"x\"><parameter name=\"y\">1</parameter></invoke></function_calls>世界"
        clean = brain._clean_xml_artifacts(dirty)
        for tag in ("function_calls", "invoke", "parameter"):
            assert tag not in clean

    def test_clean_text_unchanged(self):
        brain, _ = make_brain()
        assert brain._clean_xml_artifacts("正常文本") == "正常文本"


# ============================================================
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
