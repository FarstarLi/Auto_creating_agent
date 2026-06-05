"""
create_workflow — 将复杂任务拆分为多步工作流,后台执行.
LLM 在发现任务无法一次完成时调用此工具.
"""

import json


def create_workflow(name: str, description: str, steps_json: str) -> str:
    """
    创建一个后台工作流.

    参数:
        name: 工作流名称
        description: 工作流用途描述
        steps_json: JSON 步骤列表,每步含 tool 和 args
                    例: [{"tool":"run_command","args":{"command":"curl..."}},
                          {"tool":"write_file","args":{"path":"out.txt","content":"..."}}]

    返回:
        工作流创建结果
    """
    try:
        steps = json.loads(steps_json)
    except json.JSONDecodeError as e:
        return f"错误: steps_json 格式错误 — {e}"

    if not isinstance(steps, list) or len(steps) == 0:
        return "错误: steps_json 必须是非空列表"

    for i, step in enumerate(steps):
        if not isinstance(step, dict) or "tool" not in step:
            return f"错误: 步骤{i+1}缺少 tool 字段"

    from tools.workflow import WorkflowEngine
    engine = WorkflowEngine()

    wf = engine.create(
        name=name,
        description=description,
        steps=steps,
    )

    from tools.mcp_pool import MCPToolPool
    pool = MCPToolPool(
        pool_file="./tools/mcp_tools.json",
        code_dir="./tools/tool_add/tool_direct",
    )
    engine.run_async(wf["id"], pool)

    return (
        f"✅ 工作流已创建并后台执行\n"
        f"   ID: {wf['id']}\n"
        f"   名称: {name}\n"
        f"   步骤: {len(steps)} 步\n"
        f"   查看进度: 调用 check_workflow('{wf['id']}')"
    )