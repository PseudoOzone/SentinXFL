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

# RAG depends on chromadb which may not be installed
try:
    from sentinxfl.llm.rag import (
        RAGPipeline,
        Document,
        RetrievalResult,
        RAGResponse,
        FraudKnowledgeBase,
    )
except ImportError:
    # Provide lightweight stubs when chromadb is unavailable
    from dataclasses import dataclass, field
    from typing import Any

    @dataclass
    class Document:
        id: str
        content: str
        metadata: dict[str, Any] = field(default_factory=dict)
        embedding: list[float] | None = None

    @dataclass
    class RetrievalResult:
        documents: list[Document] = field(default_factory=list)
        distances: list[float] = field(default_factory=list)
        query: str = ""

    @dataclass
    class RAGResponse:
        answer: str = ""
        sources: list[Document] = field(default_factory=list)

    RAGPipeline = None  # type: ignore
    FraudKnowledgeBase = None  # type: ignore

# Explainer also depends on rag indirectly
try:
    from sentinxfl.llm.explainer import (
        FraudExplainer,
        FraudExplanation,
        FeatureContribution,
        ExplanationType,
        ExplanationConfig,
        create_explainer,
    )
except ImportError:
    FraudExplainer = None  # type: ignore
    FraudExplanation = None  # type: ignore
    FeatureContribution = None  # type: ignore
    ExplanationType = None  # type: ignore
    ExplanationConfig = None  # type: ignore
    create_explainer = None  # type: ignore

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
