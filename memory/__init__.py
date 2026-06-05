"""memory 模块 — 三层对话记忆系统"""

from .models import (
    Memory,
    LongTermMemory,
    Message,
    MemoryItem,
    estimate_tokens,
    estimate_messages_tokens,
)
from .utils import handle_memory_overflow, summarize_func, load_memories_for_conversation
from .cleanup import delete_least_important_summaries
from .manager import MemoryManager
from .embeddings import create_embedder, TFIDFEmbedder, OpenAIEmbedder, cosine_similarity

__all__ = [
    # 数据模型
    "Memory",
    "LongTermMemory",
    "Message",
    "MemoryItem",
    # Token
    "estimate_tokens",
    "estimate_messages_tokens",
    # 兼容旧 API
    "handle_memory_overflow",
    "summarize_func",
    "load_memories_for_conversation",
    "delete_least_important_summaries",
    # 三层记忆管理器
    "MemoryManager",
    # Embedding
    "create_embedder",
    "TFIDFEmbedder",
    "OpenAIEmbedder",
    "cosine_similarity",
]
