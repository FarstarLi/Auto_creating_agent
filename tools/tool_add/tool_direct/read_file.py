def read_file(path: str) -> str:
    """读取本地文件内容"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败：{e}"
