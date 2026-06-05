import os

def create_directory(path: str) -> str:
    """创建目录，如果目录不存在则创建，支持多级目录"""
    try:
        os.makedirs(path, exist_ok=True)
        return f"目录创建成功：{path}"
    except Exception as e:
        return f"创建目录失败：{e}"
