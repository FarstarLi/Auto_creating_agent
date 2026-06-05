def write_file(path: str, content: str, mode: str = "w") -> str:
    """
    写入文件内容
    :param path: 文件路径
    :param content: 要写入的内容
    :param mode: 写入模式，'w'覆盖，'a'追加，默认为'w'
    :return: 操作结果字符串
    """
    try:
        if mode not in ("w", "a"):
            return f"错误：不支持的写入模式 '{mode}'，请使用 'w' 或 'a'"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        action = "覆盖写入" if mode == "w" else "追加写入"
        return f"文件{action}成功：{path}"
    except Exception as e:
        return f"写入文件失败：{e}"
