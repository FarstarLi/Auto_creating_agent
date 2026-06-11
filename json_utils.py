"""
json_utils —— 健壮的 LLM 输出 JSON 解析工具（brain 和 memory 共用）。

LLM 经常不按规定输出纯 JSON：散文混 JSON、markdown 围栏包裹、
content 为 None 等。本模块提供绝不抛异常的提取函数，按序降级：

1. None/空 → None
2. 直接 json.loads
3. 剥 markdown 代码围栏后 loads
4. 平衡括号扫描提取首个完整 JSON 对象
5. 全部失败 → None
"""

import json
import re
from typing import Optional, Tuple

# markdown 代码围栏：```json ... ``` 或 ``` ... ```
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE)


def _loads_dict(text: str) -> Optional[dict]:
    """尝试 json.loads，成功且结果为 dict 才返回"""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _scan_first_object(text: str) -> Optional[dict]:
    """平衡括号扫描：提取文本中首个完整的 JSON 对象。

    正确处理字符串内的 {} 与转义引号。
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    data = _loads_dict(candidate)
                    if data is not None:
                        return data
                    break  # 该候选不合法，从下一个 { 重试
        start = text.find("{", start + 1)
    return None


def extract_json(text) -> Optional[dict]:
    """从 LLM 输出中提取首个 JSON 对象。失败返回 None，绝不抛异常。"""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    # 1) 直接解析
    data = _loads_dict(text)
    if data is not None:
        return data

    # 2) 剥 markdown 围栏
    m = _FENCE_RE.search(text)
    if m:
        data = _loads_dict(m.group(1).strip())
        if data is not None:
            return data
        # 围栏内可能仍混有散文，继续扫描围栏内容
        data = _scan_first_object(m.group(1))
        if data is not None:
            return data

    # 3) 平衡括号扫描全文
    return _scan_first_object(text)


def parse_llm_json(text, expected_keys: Tuple[str, ...] = ()) -> Optional[dict]:
    """提取并校验 LLM 返回的 JSON。

    Args:
        text: LLM 输出文本
        expected_keys: 期望键（至少命中一个才有效，防止提取到无关的内嵌 JSON）

    Returns:
        合法 dict 或 None
    """
    data = extract_json(text)
    if data is None:
        return None
    if expected_keys and not any(k in data for k in expected_keys):
        return None
    return data
