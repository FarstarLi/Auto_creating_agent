"""模型适配器 —— 多模型统一接口"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    SILICONFLOW = "siliconflow"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """模型配置"""
    provider: ModelProvider
    model_name: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseModelAdapter(ABC):
    """模型适配器基类"""

    @abstractmethod
    def chat(self, messages: List[Dict], tools: Optional[List] = None,
             json_mode: bool = False) -> Dict:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass


class OpenAIAdapter(BaseModelAdapter):
    """OpenAI 兼容适配器 (支持 SiliconFlow, DeepSeek 等)"""

    def __init__(self, config: ModelConfig):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url or "https://api.openai.com/v1",
            timeout=120,
        )
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        # 实例级能力缓存：后端不支持 response_format 时降级并不再尝试
        self._json_mode_ok = True

    @staticmethod
    def _ensure_json_keyword(messages: List[Dict]) -> List[Dict]:
        """DeepSeek JSON mode 要求 prompt 中含 'json' 字样，缺则补充"""
        for m in messages:
            if "json" in str(m.get("content", "")).lower():
                return messages
        messages = list(messages)
        if messages and messages[-1].get("role") == "user":
            last = dict(messages[-1])
            last["content"] = str(last.get("content", "")) + "\n（请输出 json 格式）"
            messages[-1] = last
        else:
            messages.append({"role": "user", "content": "请输出 json 格式"})
        return messages

    def chat(self, messages: List[Dict], tools: Optional[List] = None,
             json_mode: bool = False) -> Dict:
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            params["tools"] = tools
        # 注意：response_format json_object 与 tools 互斥（DeepSeek），
        # 仅在无 tools 的纯文本调用（THINK/REFLECT）启用
        if json_mode and self._json_mode_ok and not tools:
            params["messages"] = self._ensure_json_keyword(messages)
            params["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**params)
        except Exception:
            if "response_format" not in params:
                raise
            # 兼容后端不支持 response_format → 降级重发，之后不再尝试
            self._json_mode_ok = False
            params.pop("response_format", None)
            params["messages"] = messages
            response = self.client.chat.completions.create(**params)
        msg = response.choices[0].message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [{
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in msg.tool_calls]

        return {
            "content": msg.content,
            "tool_calls": tool_calls,
            "finish_reason": response.choices[0].finish_reason,
        }

    def get_model_name(self) -> str:
        return self.model_name


class OllamaAdapter(BaseModelAdapter):
    """Ollama 本地模型适配器"""

    def __init__(self, config: ModelConfig):
        import requests
        self.base_url = config.base_url or "http://localhost:11434"
        self.model_name = config.model_name
        self.session = requests.Session()

    def chat(self, messages: List[Dict], tools: Optional[List] = None,
             json_mode: bool = False) -> Dict:
        payload = {"model": self.model_name, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"  # Ollama 原生 JSON mode
        response = self.session.post(f"{self.base_url}/api/chat", json=payload)
        if not response.ok:
            return {
                "content": f"[Ollama 错误 {response.status_code}] {response.text[:300]}",
                "tool_calls": None,
                "finish_reason": "error",
            }
        data = response.json()
        return {
            "content": data.get("message", {}).get("content", ""),
            "tool_calls": None,
            "finish_reason": "stop",
        }

    def get_model_name(self) -> str:
        return self.model_name


def create_model_adapter(config: ModelConfig) -> BaseModelAdapter:
    """工厂函数：根据 provider 创建对应的适配器"""
    if config.provider == ModelProvider.OLLAMA:
        return OllamaAdapter(config)
    return OpenAIAdapter(config)
