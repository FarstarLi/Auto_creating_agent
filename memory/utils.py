import hashlib
from datetime import datetime
from typing import Callable, Optional
from memory.models import Memory,MemoryItem,LongTermMemory,Message
import os
from openai import OpenAI

def handle_memory_overflow(
    memory: Memory,
    long_term_memory: LongTermMemory,
    summarize_func: Callable[[str,OpenAI,str], str],
    client:OpenAI,
    model:str="gpt-3.5-turbo",
    archive_dir: str = "./memory/archives",
    archive_prefix: str = "long_term_archive",
    raw_tag: str = "raw_message",
    summary_tag: str = "summary"
) -> None:
    """
    处理短期记忆溢出：
    1. 当消息数量超过 max_messages 时：
       a. 将前 80% 的消息逐条存入长期记忆
       b. 对这些消息进行 AI 总结
    2. 截断短期记忆，保留最后 20%
    3. 长期记忆满了则存档并清空

    Args:
        memory: 短期记忆实例
        long_term_memory: 长期记忆实例
        summarize_func: AI 总结函数，接受一段文本返回摘要字符串
        archive_dir: 长期记忆存档目录
        archive_prefix: 存档文件前缀
        raw_tag: 原始消息的标签
        summary_tag: 总结的标签
    """
    if len(memory) < memory.max_messages:
        print('未超出，不给予总结')
        return
    print('正在处理')
    total = len(memory.messages)
    delete_count = int(total * 0.8)
    keep_count = total - delete_count

    messages_to_archive = memory.messages[:delete_count]

    # ========== 1a. 将每条原始消息逐条存入长期记忆 ==========
    for msg in messages_to_archive:
        content_str = f"[{msg.role}] {msg.content or ''}"
        raw_memory_item = MemoryItem(
            content=content_str,
            importance=0.5, 
            tags=[raw_tag, f"role:{msg.role}"],
            metadata={
                "archived_at": datetime.now().isoformat(),
                "original_message": msg.to_dict()  
            }
        )
        long_term_memory.add_item(raw_memory_item)

    # ========== 1b. AI 总结并存入长期记忆 ==========
    summary_input = "\n".join(
        f"[{msg.role}] {msg.content or ''}" for msg in messages_to_archive
    )
    print(f'这是要总结的内容：{summary_input}')
    try:
        summary_text = summarize_func(summary_input,client,model)
    except Exception as e:
        print(f"AI 总结失败: {e}")
        summary_text = "[总结失败]"
    print(f'这是已经总结的内容：{summary_text}')
    summary_memory_item = MemoryItem(
        content=summary_text,
        importance=0.8,
        tags=[summary_tag, "conversation_archive"],
        metadata={
            "summarized_at": datetime.now().isoformat(),
            "source_message_count": len(messages_to_archive)
        }
    )
    long_term_memory.add_item(summary_memory_item)
    # ========== 3. 检查长期记忆容量，满则存档并清空 ==========
    if long_term_memory.max_items is not None and len(long_term_memory) >= long_term_memory.max_items:
        os.makedirs(archive_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_filepath = os.path.join(archive_dir, f"{archive_prefix}.json")
        long_term_memory.save_to_file(archive_filepath)
        long_term_memory.clear()
        print(f"长期记忆已存档至: {archive_filepath}")

    kept_messages = memory.messages[-keep_count:]
    summary_message = Message.system_message(content=summary_text)
    memory.messages = [summary_message] + kept_messages
    print("总结成功")

def summarize_func(
    text: str,
    client: OpenAI,
    model: str = "gpt-3.5-turbo"  # 默认模型，可覆盖
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个对话总结助手，请用简洁的语言总结以下对话的核心内容，保留重要信息。(主要形式为：事件与当前进度)"},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        summary = response.choices[0].message.content.strip()
        return summary if summary else "[无法生成摘要]"
    except Exception as e:
        print(f"AI 总结失败: {e}")
        return "[总结失败]"

import os
from typing import Optional
from memory.models import Memory,LongTermMemory,Message
def load_memories_for_conversation(
    short_term_memory_file: str,
    long_term_memory_file: str,
    summary_tag: str = "summary",
) -> Memory:
    """
    从短期记忆文件和长期记忆文件中加载数据，合并到短期记忆中。
    合并顺序：先添加所有摘要消息（按时间升序），再添加短期记忆消息。
    如果总消息数超过 max_messages，则截断最早的消息（保留最新的 max_messages 条）。

    Args:
        short_term_memory_file: 短期记忆文件路径（JSON 格式）
        long_term_memory_file: 长期记忆文件路径（JSON 格式）
        summary_tag: 用于识别摘要内容的标签，默认为 "summary"
        max_messages: 短期记忆的最大消息数

    Returns:
        Memory: 合并后的短期记忆实例
    """
    # 1. 加载短期记忆（如果文件存在）
    if os.path.exists(short_term_memory_file) and os.path.getsize(short_term_memory_file) > 0 :
        short_memory = Memory.load_from_file(short_term_memory_file)
    else:
        short_memory = Memory()

    # 2. 加载长期记忆（如果文件存在），提取摘要
    summary_messages = []
    if os.path.exists(long_term_memory_file) and os.path.getsize(long_term_memory_file) > 0:
        long_term = LongTermMemory.load_from_file(long_term_memory_file)
        summary_items = sorted(
            [item for item in long_term.items if summary_tag in item.tags],
            key=lambda x: x.timestamp
        )
        if len(summary_items) != 0:
            for item in summary_items:
                time_str = item.timestamp.strftime("%Y-%m-%d %H:%M")
                content = f"[Summary from {time_str}] {item.content}"
                summary_messages.append(Message.system_message(content=content))

    merged_messages = summary_messages + short_memory.messages

    result_memory = Memory(messages=merged_messages)

    return result_memory
