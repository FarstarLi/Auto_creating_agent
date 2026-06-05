"""
check_workflow — 查看工作流状态和结果.
"""


def check_workflow(wf_id: str = "") -> str:
    """
    查询工作流状态.

    参数:
        wf_id: 工作流 ID(为空则列出所有工作流)

    返回:
        工作流状态或列表
    """
    from tools.workflow import WorkflowEngine
    engine = WorkflowEngine()

    # 列出所有
    if not wf_id:
        all_wf = engine.list_workflows()
        if not all_wf:
            return "📋 当前无工作流"
        lines = [engine.summary(), ""]
        for wf in all_wf[:10]:
            lines.append(
                f"  [{wf['status']}] {wf['id']} — {wf['name']} "
                f"({wf.get('current_step',0)}/{len(wf.get('steps',[]))}步)"
            )
        return "\n".join(lines)

    # 查询单个
    wf = engine.get(wf_id)
    if not wf:
        return f"错误: 工作流 '{wf_id}' 不存在"

    return engine._format_result(wf)