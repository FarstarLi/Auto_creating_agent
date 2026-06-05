import os

def delete_file(path: str) -> str:
    """删除指定文件"""
    try:
        if not os.path.exists(path):
            return f"错误：文件不存在，无法删除：{path}"
        os.remove(path)
        return f"文件删除成功：{path}"
    except Exception as e:
        return f"删除文件失败：{e}"
