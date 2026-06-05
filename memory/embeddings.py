"""
记忆 Embedding 模块 — 文本向量化 + 语义检索。

支持:
- OpenAI text-embedding-3-small (需 API key)
- 本地 TF-IDF 回退（无需 API）
- 余弦相似度检索
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ==================== 通用工具 ====================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(v * v for v in vec1)) or 1.0
    norm2 = math.sqrt(sum(v * v for v in vec2)) or 1.0
    return dot / (norm1 * norm2)


class TFIDFEmbedder:
    """轻量级本地 TF-IDF embedder，无需 API，开箱即用"""

    def __init__(self):
        self._doc_count = 0
        self._df: Dict[str, int] = defaultdict(int)  # document frequency

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词：中文按字，英文按词"""
        tokens = []
        # 中文单字
        chinese = re.findall(r"[一-鿿]", text)
        tokens.extend(chinese)
        # 英文/数字词
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens.extend(words)
        return tokens

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """计算词频"""
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        # L2 归一化
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        return {k: v / norm for k, v in tf.items()}

    def embed(self, text: str) -> Tuple[List[float], Dict[str, float]]:
        """
        返回 (dense_vector, sparse_tf)。

        TF-IDF 没有真正的 dense vector，这里返回稀疏 TF 向量，
        同时用于构建检索索引。
        """
        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)
        # 更新 DF
        for t in set(tokens):
            self._df[t] += 1
        self._doc_count += 1
        # 生成一个简单的 dense 表示（基于 TF-IDF 权重）
        vec = list(tf.values())
        return vec, tf

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度"""
        return cosine_similarity(vec1, vec2)

    def search_by_tfidf(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        基于 TF-IDF 的关键词检索。

        Args:
            query: 查询文本
            documents: 文档列表，每项需含 'content' 字段
            top_k: 返回条数

        Returns:
            [(document, score), ...] 按分数降序
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(d, 0.0) for d in documents[:top_k]]

        # 计算 query 的 TF
        query_tf: Dict[str, float] = defaultdict(float)
        for t in query_tokens:
            query_tf[t] += 1.0

        scores = []
        for doc in documents:
            content = doc.get("content", "")
            doc_tokens = self._tokenize(content)
            if not doc_tokens:
                scores.append((doc, 0.0))
                continue
            # BM25 简化版: TF * IDF
            score = 0.0
            for t in set(query_tokens):
                if t in doc_tokens:
                    tf = doc_tokens.count(t) / len(doc_tokens)
                    df = self._df.get(t, 1)
                    idf = math.log((self._doc_count + 1) / (df + 1)) + 1.0
                    score += tf * idf * query_tf[t]
            scores.append((doc, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small 包装"""

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model
        self._dim = 1536

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        """生成 embedding 向量"""
        try:
            resp = self.client.embeddings.create(model=self.model, input=text)
            return resp.data[0].embedding
        except Exception:
            logger.debug("embed error", exc_info=True)
            return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成 embedding"""
        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            logger.debug("embed error", exc_info=True)
            return [[] for _ in texts]

    @staticmethod
    def similarity(vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度"""
        return cosine_similarity(vec1, vec2)

    def search(
        self,
        query_vec: List[float],
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        向量相似度检索。

        Args:
            query_vec: 查询的 embedding
            documents: 文档列表，每项需含 'embedding' 字段
            top_k: 返回条数

        Returns:
            [(document, score), ...] 按分数降序
        """
        if not query_vec:
            return [(d, 0.0) for d in documents[:top_k]]

        scores = []
        for doc in documents:
            doc_vec = doc.get("embedding", [])
            sim = self.similarity(query_vec, doc_vec)
            scores.append((doc, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==================== 统一 Embedder 工厂 ====================

def create_embedder(client=None, model: str = "text-embedding-3-small") -> Any:
    """
    创建 embedder 实例。
    优先使用 OpenAI（如果提供 client），否则使用本地 TF-IDF。
    """
    if client is not None:
        try:
            return OpenAIEmbedder(client, model)
        except Exception:
            logger.debug("create embedder failed", exc_info=True)
            return TFIDFEmbedder()
    return TFIDFEmbedder()
