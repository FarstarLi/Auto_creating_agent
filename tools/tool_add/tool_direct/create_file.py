import os

def create_file(path: str, content: str = "", overwrite: bool = False) -> str:
    """
    创建文件并写入初始内容
    :param path: 文件路径
    :param content: 初始内容，默认为空字符串
    :param overwrite: 是否覆盖已存在的文件，默认为False（不覆盖）
    :return: 操作结果字符串
    """
    try:
        if os.path.exists(path) and not overwrite:
            return f"错误：文件已存在且 overwrite=False，无法创建：{path}"
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件创建成功：{path}"
    except Exception as e:
        return f"创建文件失败：{e}"
