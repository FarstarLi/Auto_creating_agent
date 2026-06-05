"""
LLM 自主创建工具集成测试 —— 验证 LLM 调用 create_tool 创建、使用、管理工具

运行方式（先在 config.json 填写 API Key）:
    cd c:/Users/qq215/Desktop/auto_coding
    python test/test_llm_create_tool.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from tools.mcp_pool import MCPToolPool
from brain.call import call_openai_with_tools

# 从 config.json 读取配置
client = config.create_llm_client()
DEEPSEEK_MODEL = config.llm_model

pool = MCPToolPool(
    pool_file="./tools/mcp_tools.json",
    code_dir="./tools/tool_add/tool_direct",
)

# 清理可能残留的测试工具
for name in ["get_weather", "reverse_string", "count_words", "fetch_title"]:
    if pool.has_tool(name):
        pool.delete_tool(name)

print(f"📦 初始 MCP 池: {pool.get_tool_count()}")
print(f"   工具列表: {list(pool.tools.keys())}")
print()


# ==================== 工具函数 ====================
def chat_with_tools(messages, max_rounds=8):
    """
    与 LLM 对话，支持多轮工具调用。
    LLM 可以自主决定调用 create_tool 或其他已有工具。
    """
    for round_idx in range(max_rounds):
        tools = pool.list_tools()

        response = call_openai_with_tools(messages, tools, client=client, model=DEEPSEEK_MODEL)
        msg = response

        # 无工具调用 → LLM 直接回复
        if not msg.tool_calls:
            return msg.content, round_idx + 1

        # 有工具调用
        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            print(f"  🔧 第{round_idx+1}轮: {func_name}({json.dumps(args, ensure_ascii=False)[:100]})")

            result = pool.execute(func_name, args)
            print(f"  📋 返回: {result[:120]}{'...' if len(result) > 120 else ''}")

            if func_name == "create_tool" and "成功" in result:
                print(f"  🆕 LLM 自主创建了新工具！")

            # 把工具调用和结果加入消息历史
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "[达到最大轮次，未完成]", max_rounds


# ============================================================
# 场景 1: LLM 发现缺少工具 → 自主创建 → 立即使用
# ============================================================
print("=" * 60)
print("  场景 1: LLM 自主创建工具")
print("=" * 60)

# 确认池中没有 joke 相关工具
assert not pool.has_tool("get_joke"), "池中不应有 get_joke"
assert not pool.has_tool("tell_joke"), "池中不应有 tell_joke"
print("❌ 池中无讲笑话的工具\n")

# 记录创建前的工具数量
tools_before = set(pool.tools.keys())

messages = [
    {
        "role": "system",
        "content": (
            "你是一个智能助手，可以使用工具完成任务。"
            "MCP 池中有许多工具。如果找不到合适的工具，必须使用 create_tool 创建。"
            "创建工具时编写完整可运行的 Python 代码，返回类型为 str。"
            "创建成功后立即调用新工具完成任务。"
            "不要跳过创建步骤直接回答——如果没有工具就必须创建。"
        ),
    },
    {
        "role": "user",
        "content": "请讲一个关于编程的笑话。如果没有对应工具，请先创建再调用。",
    },
]

answer, rounds = chat_with_tools(messages)
print(f"\n💬 最终回复 ({rounds}轮): {answer}")
print()

# 检查是否有新工具被创建
tools_after = set(pool.tools.keys())
new_tools = tools_after - tools_before
if new_tools:
    print(f"✅ LLM 成功创建了新工具: {new_tools}")
    created_tool_name = list(new_tools)[0]
else:
    created_tool_name = None
    print("⚠️ LLM 没有创建新工具（可能直接回答了）")


# ============================================================
# 场景 2: LLM 创建工具 → 再次对话直接使用（持久化验证）
# ============================================================
print("\n" + "=" * 60)
print("  场景 2: 持久化后直接使用已创建工具")
print("=" * 60)

if created_tool_name and pool.has_tool(created_tool_name):
    print(f"✅ {created_tool_name} 已在池中，LLM 应该直接复用\n")

    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手。优先使用已有工具，不要重复创建。",
        },
        {
            "role": "user",
            "content": "再讲一个关于程序员的冷笑话。",
        },
    ]

    answer, rounds = chat_with_tools(messages)
    print(f"\n💬 最终回复 ({rounds}轮): {answer}")
else:
    created_tool_name = None
    print("⚠️ 跳过（场景1未创建工具）")


# ============================================================
# 场景 3: LLM 使用已有工具完成计算任务
# ============================================================
print("\n" + "=" * 60)
print("  场景 3: LLM 正确选择已有工具")
print("=" * 60)

print(f"📦 池中有 compute_factorial: {pool.has_tool('compute_factorial')}")
print()

messages = [
    {
        "role": "system",
        "content": "你是一个智能助手。使用已有工具完成任务。不要创建重复工具。",
    },
    {
        "role": "user",
        "content": "请帮我计算 10 的阶乘。",
    },
]

answer, rounds = chat_with_tools(messages)
print(f"\n💬 最终回复 ({rounds}轮): {answer}")


# ============================================================
# 场景 4: LLM 搜索工具（通过 search_mcp_tools 等）
# ============================================================
print("\n" + "=" * 60)
print("  场景 4: 查看 MCP 池最终状态")
print("=" * 60)

print(pool.summary())

# 清理
if created_tool_name and pool.has_tool(created_tool_name):
    pool.delete_tool(created_tool_name)
    print(f"\n🧹 已清理 {created_tool_name}")
