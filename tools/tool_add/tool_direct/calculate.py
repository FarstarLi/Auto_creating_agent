import operator, math

_ops = {
    "**": operator.pow, "^": operator.pow,
    "*": operator.mul, "/": operator.truediv,
    "+": operator.add, "-": operator.sub,
    "%": operator.mod, "//": operator.floordiv,
}

def calculate(expr: str) -> str:
    """安全计算数学表达式，支持 + - * / ** % // 和括号"""
    try:
        result = eval(expr, {"__builtins__": {}}, {
            "pow": pow, "abs": abs, "round": round,
            "max": max, "min": min, "int": int, "float": float,
            "math": math,
        })
        return str(result)
    except Exception as e:
        return f"计算失败: {e}"
