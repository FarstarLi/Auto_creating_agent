import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage


def call_openai_with_tools(
    messages: List[Dict[str, Any]], tools: List[Dict], client, model: str = "gpt-4o"
) -> ChatCompletionMessage:
    """
    调用 OpenAI ChatCompletion 并返回响应消息（可能包含工具调用）。

    Args:
        messages: 对话历史消息列表
        tools: 工具描述列表（OpenAI 格式，从 MCPToolPool.list_tools() 获取）
        model: 使用的模型名称，默认 "gpt-4o"

    Returns:
        ChatCompletionMessage 对象，包含 assistant 的回复内容或工具调用。
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        return response.choices[0].message
    except Exception as e:
        logging.error(f"OpenAI API 调用失败: {e}")
        return ChatCompletionMessage(
            role="assistant",
            content=f"调用 OpenAI API 时发生错误: {e}",
            tool_calls=None
        )


def execute_tool_call(tool_call, pool=None) -> str:
    """
    执行单个工具调用，返回执行结果字符串。

    Args:
        tool_call: 来自 OpenAI 响应的工具调用对象
        pool: MCPToolPool 实例（推荐）。如果为 None，回退到动态 import。

    Returns:
        工具执行结果的字符串表示。
    """
    func_name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        arguments = {}

    # 使用 MCP 池执行
    if pool is not None:
        return pool.execute(func_name, arguments)

    # 回退：尝试从 tool_add 动态导入
    import importlib
    try:
        module = importlib.import_module(f'tools.tool_add.tool_direct.{func_name}')
        func = getattr(module, func_name)
        result = func(**arguments)
        return str(result)
    except Exception as e:
        return f"工具执行异常：{e}"
