"""
SentinXFL - LLM Provider Interface
===================================

Abstract interface for LLM providers with implementations
for Ollama (free/local), OpenAI, Anthropic, and Groq.

Author: Anshuman Bakshi
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    
    OLLAMA = "ollama"
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = "llama3.2"
    
    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    
    # API settings
    api_key: str | None = None
    api_base: str = "http://localhost:11434"
    timeout: int = 60
    
    # System prompt
    system_prompt: str = """You are a fraud detection expert assistant for SentinXFL.
You help bank analysts understand fraud patterns, model predictions, and
provide actionable insights. Be concise, accurate, and focus on financial security."""


@dataclass
class LLMResponse:
    """Response from LLM."""
    
    content: str
    model: str
    provider: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Chat message for conversation."""
    
    role: str  # "system", "user", "assistant"
    content: str


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (check connectivity, load model, etc.)."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> LLMResponse:
        """Generate response in chat format."""
        pass
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream text generation."""
        pass
    
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
    
    async def is_available(self) -> bool:
        """Check if provider is available for use."""
        if not self._initialized:
            try:
                await self.initialize()
            except Exception:
                return False
        return self._initialized
    
    async def health_check(self) -> dict[str, Any]:
        """Check provider health."""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "initialized": self._initialized,
        }


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM inference.
    
    FREE - runs entirely on local hardware.
    Supports: Llama 3.2, Mistral, CodeLlama, etc.
    """
    
    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self.client = None
        
        # Default to Ollama settings
        if self.config.provider != LLMProvider.OLLAMA:
            self.config.provider = LLMProvider.OLLAMA
        
        if not self.config.api_base:
            self.config.api_base = "http://localhost:11434"
    
    async def initialize(self) -> None:
        """Initialize Ollama connection."""
        try:
            import httpx
            
            # Check if Ollama is running
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.config.api_base}/api/tags")
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    
                    # Check if required model is available
                    if not any(self.config.model in name for name in model_names):
                        logger.warning(
                            f"Model {self.config.model} not found. "
                            f"Available: {model_names}. "
                            f"Run: ollama pull {self.config.model}"
                        )
                    
                    self._initialized = True
                    logger.info(f"Ollama initialized with model {self.config.model}")
                else:
                    raise ConnectionError("Ollama not responding")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            self._initialized = False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate text using Ollama."""
        import httpx
        import time
        
        system = system_prompt or self.config.system_prompt
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.api_base}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "system": system,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", self.config.temperature),
                            "top_p": kwargs.get("top_p", self.config.top_p),
                            "top_k": kwargs.get("top_k", self.config.top_k),
                            "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                        },
                    },
                )
                
                data = response.json()
                latency = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    content=data.get("response", ""),
                    model=self.config.model,
                    provider="ollama",
                    tokens_used=data.get("eval_count", 0),
                    latency_ms=latency,
                    metadata={
                        "total_duration": data.get("total_duration"),
                        "load_duration": data.get("load_duration"),
                    },
                )
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.config.model,
                provider="ollama",
            )
    
    async def chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> LLMResponse:
        """Chat completion using Ollama."""
        import httpx
        import time
        
        start_time = time.time()
        
        # Convert to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self.config.api_base}/api/chat",
                    json={
                        "model": self.config.model,
                        "messages": ollama_messages,
                        "stream": False,
                        "options": {
                            "temperature": kwargs.get("temperature", self.config.temperature),
                            "top_p": kwargs.get("top_p", self.config.top_p),
                            "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                        },
                    },
                )
                
                data = response.json()
                latency = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=self.config.model,
                    provider="ollama",
                    tokens_used=data.get("eval_count", 0),
                    latency_ms=latency,
                )
                
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            return LLMResponse(
                content=f"Error: {str(e)}",
                model=self.config.model,
                provider="ollama",
            )
    
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generation from Ollama."""
        import httpx
        
        system = system_prompt or self.config.system_prompt
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.config.api_base}/api/generate",
                    json={
                        "model": self.config.model,
                        "prompt": prompt,
                        "system": system,
                        "stream": True,
                        "options": {
                            "temperature": kwargs.get("temperature", self.config.temperature),
                            "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                        },
                    },
                ) as response:
                    import json
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                                
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            yield f"Error: {str(e)}"
    
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings using Ollama."""
        import httpx
        
        embeddings = []
        
        # Ollama embedding model
        embed_model = "nomic-embed-text"
        
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                for text in texts:
                    response = await client.post(
                        f"{self.config.api_base}/api/embeddings",
                        json={
                            "model": embed_model,
                            "prompt": text,
                        },
                    )
                    
                    data = response.json()
                    embeddings.append(data.get("embedding", []))
                    
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            # Return zero vectors on failure
            embeddings = [[0.0] * 768 for _ in texts]
        
        return embeddings


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider for testing.
    
    Returns predefined responses without external API calls.
    """
    
    async def initialize(self) -> None:
        """Initialize mock provider."""
        self._initialized = True
        logger.info("Mock LLM provider initialized")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate mock response."""
        return LLMResponse(
            content=f"Mock response for: {prompt[:50]}...",
            model="mock",
            provider="mock",
            tokens_used=len(prompt.split()),
            latency_ms=10.0,
        )
    
    async def chat(
        self,
        messages: list[ChatMessage],
        **kwargs,
    ) -> LLMResponse:
        """Generate mock chat response."""
        last_msg = messages[-1].content if messages else ""
        return LLMResponse(
            content=f"Mock chat response for: {last_msg[:50]}...",
            model="mock",
            provider="mock",
        )
    
    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream mock response."""
        for word in f"Mock streaming response for {prompt[:20]}".split():
            yield word + " "
    
    async def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate mock embeddings."""
        import numpy as np
        return [np.random.randn(768).tolist() for _ in texts]


# ==========================================
# Provider Factory
# ==========================================

_provider_instance: BaseLLMProvider | None = None


def get_llm_provider(
    config: LLMConfig | None = None,
) -> BaseLLMProvider:
    """
    Get LLM provider instance.
    
    Args:
        config: Provider configuration
        
    Returns:
        LLM provider instance
    """
    global _provider_instance
    
    # Create default config if not provided
    if config is None:
        config = LLMConfig(
            provider=LLMProvider(settings.llm_provider),
            model=settings.llm_model_id,
            api_base=settings.ollama_url,
        )
    
    # Determine provider type
    provider_str = config.provider.value
    
    # Create provider
    if provider_str == "ollama" or provider_str == "local":
        return OllamaProvider(config)
    elif provider_str == "mock":
        return MockLLMProvider(config)
    else:
        # Default to Ollama for free usage
        logger.warning(f"Provider {provider_str} not implemented, using Ollama")
        return OllamaProvider(config)


async def get_initialized_provider(
    provider: LLMProvider | str | None = None,
) -> BaseLLMProvider:
    """Get an initialized LLM provider."""
    llm = get_llm_provider(provider)
    
    if not llm.is_initialized():
        await llm.initialize()
    
    return llm
