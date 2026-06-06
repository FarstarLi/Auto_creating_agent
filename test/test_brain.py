"""
智能体大脑测试脚本 v2.2
测试四状态机 + MCP 工具池 + 三层记忆
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from tools.mcp_pool import MCPToolPool


# ============================================================
# 测试 0: MCP 工具池基础功能（不涉及 LLM）
# ============================================================
def test_mcp_pool_basics():
    """测试 MCPToolPool 的加载、搜索、执行、注册"""
    print("\n" + "="*60)
    print("测试 0: MCP 工具池基础功能")
    print("="*60)

    pool = MCPToolPool(
        pool_file=config.tools_pool_file,
        code_dir=config.tools_code_dir,
    )

    # 1. 加载
    stats = pool.get_tool_count()
    print(f"✓ 加载工具: {stats}")
    assert stats["total"] >= 7, f"至少应有7个核心工具，实际: {stats['total']}"

    # 2. 搜索
    results = pool.search("文件")
    print(f"✓ 搜索 '文件': {[r['name'] for r in results]}")
    assert len(results) >= 2, f"应能找到至少2个文件相关工具"

    # 3. 执行
    result = pool.execute("read_file", {"path": "./README.md"})
    print(f"✓ read_file 执行: {result[:60]}...")
    assert len(result) > 0

    # 4. 创建并执行工具
    new_code = '''
def compute_factorial(N: int) -> str:
    """计算阶乘"""
    result = 1
    for i in range(1, N + 1):
        result *= i
    return f"{N}! = {result}"
'''
    result = pool.register_tool(
        name="compute_factorial",
        description="计算指定数值的阶乘",
        code=new_code,
    )
    print(f"✓ 创建 compute_factorial: {result[:80]}...")
    assert "成功" in result
    result = pool.execute("compute_factorial", {"N": 5})
    print(f"✓ compute_factorial(5): {result}")
    assert "120" in result, f"5! 应该是 120，实际: {result}"

    # 5. 创建新工具
    new_code = '''
def greet(name: str) -> str:
    """向用户打招呼"""
    return f"你好，{name}！欢迎使用 MCP 工具池。"
'''
    result = pool.register_tool(
        name="greet",
        description="向用户打招呼，传入 name 参数返回问候语",
        code=new_code,
    )
    print(f"✓ register_tool: {result[:80]}...")
    assert "成功" in result, f"创建失败: {result}"

    # 6. 立即执行新工具
    result = pool.execute("greet", {"name": "小明"})
    print(f"✓ greet('小明'): {result}")
    assert "小明" in result

    # 7. 验证 list_tools 包含 create_tool
    tools = pool.list_tools()
    tool_names = [t["function"]["name"] for t in tools]
    print(f"✓ list_tools 包含 {len(tools)} 个工具")
    assert "create_tool" in tool_names, "应包含 create_tool 元工具"

    # 8. 持久化验证
    pool2 = MCPToolPool(
        pool_file=config.tools_pool_file,
        code_dir=config.tools_code_dir,
    )
    assert pool2.has_tool("greet"), "重启后 greet 应该仍在池中"
    print("✓ 持久化验证通过: greet 在重启后仍在池中")

    # 9. 清理测试工具
    pool.delete_tool("greet")
    assert not pool.has_tool("greet")
    print("✓ 删除测试工具 greet")
    pool.delete_tool("compute_factorial")
    assert not pool.has_tool("compute_factorial")
    print("✓ 删除测试工具 compute_factorial")

    print("\n✅ 测试 0 全部通过！")
    return pool


# ============================================================
# 测试 1: AgentBrain 基础执行 + MCP 池集成（需LLM，注释掉）
# ============================================================
# def test_brain_with_mcp():
#     ...


# ============================================================
# 测试 2: LLM 自主创建工具流程（需LLM，注释掉）
# ============================================================
# def test_llm_create_tool():
#     ...


# ============================================================
if __name__ == "__main__":
    print("🧪 智能体大脑测试 v2.2")
    print("="*60)
    test_mcp_pool_basics()
