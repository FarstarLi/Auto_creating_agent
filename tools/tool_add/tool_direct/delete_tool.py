import os
import json


def delete_tool(func_name: str) -> str:
    """
    删除指定名称的工具，清理 .py 文件和 mcp_tools.json 中的条目。

    参数：
        func_name: 要删除的工具函数名
    返回：
        包含操作结果的字符串
    """
    messages = []
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    py_path = os.path.join(base_dir, "tool_direct", f"{func_name}.py")
    mcp_json_path = os.path.join(base_dir, "..", "mcp_tools.json")

    # 1. 删除 .py 文件
    try:
        if os.path.exists(py_path):
            os.remove(py_path)
            messages.append(f"已删除文件: {py_path}")
        else:
            messages.append(f"警告: 文件不存在: {py_path}")
    except Exception as e:
        messages.append(f"删除文件失败: {e}")

    # 2. 从 mcp_tools.json 中移除条目
    try:
        mcp_json_path = os.path.normpath(mcp_json_path)
        if os.path.exists(mcp_json_path):
            with open(mcp_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            new_data = [entry for entry in data if entry.get("name") != func_name]
            if len(new_data) == len(data):
                messages.append(f"警告: mcp_tools.json 中未找到 {func_name}")
            with open(mcp_json_path, "w", encoding="utf-8") as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            messages.append(f"已从 mcp_tools.json 移除 {func_name}")
        else:
            messages.append(f"警告: mcp_tools.json 不存在")
    except Exception as e:
        messages.append(f"更新 mcp_tools.json 失败: {e}")

    return "\n".join(messages)
