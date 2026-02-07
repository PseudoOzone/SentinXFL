"""LLM and RAG module for SentinXFL - Explainable AI."""

from sentinxfl.llm.provider import (
    BaseLLMProvider,
    OllamaProvider,
    MockLLMProvider,
    LLMConfig,
    LLMResponse,
    ChatMessage,
    get_llm_provider,
)
from sentinxfl.llm.rag import (
    RAGPipeline,
    Document,
    RetrievalResult,
    RAGResponse,
    FraudKnowledgeBase,
)
from sentinxfl.llm.explainer import (
    FraudExplainer,
    FraudExplanation,
    FeatureContribution,
    ExplanationType,
    ExplanationConfig,
    create_explainer,
)

__all__ = [
    # LLM Provider
    "BaseLLMProvider",
    "OllamaProvider",
    "MockLLMProvider",
    "LLMConfig",
    "LLMResponse",
    "ChatMessage",
    "get_llm_provider",
    # RAG
    "RAGPipeline",
    "Document",
    "RetrievalResult",
    "RAGResponse",
    "FraudKnowledgeBase",
    # Explainer
    "FraudExplainer",
    "FraudExplanation",
    "FeatureContribution",
    "ExplanationType",
    "ExplanationConfig",
    "create_explainer",
]
