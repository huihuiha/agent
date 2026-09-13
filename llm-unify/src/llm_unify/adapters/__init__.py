"""适配器注册表：protocol -> 适配器类。

新增一种上游协议只需在这里注册一个类，业务层与路由层零改动。
"""

from __future__ import annotations

from llm_unify.adapters.anthropic_messages import AnthropicMessagesAdapter
from llm_unify.adapters.base import ModelAdapter
from llm_unify.adapters.openai_chat import DeepSeekChatAdapter
from llm_unify.adapters.openai_responses import OpenAIResponsesAdapter

ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "openai-responses": OpenAIResponsesAdapter,
    "anthropic-messages": AnthropicMessagesAdapter,
    "openai-chat": DeepSeekChatAdapter,
}
