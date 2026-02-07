"""
Sprint 4 Tests: LLM Integration & Explainability
================================================

Tests for:
- LLM Provider (Ollama, Mock)
- RAG Pipeline (ChromaDB)
- Fraud Explainer
- LLM API Routes
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")

# ============================================================================
# LLM Provider Tests
# ============================================================================

class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_config(self):
        """Test default LLM configuration."""
        from sentinxfl.llm.provider import LLMConfig
        
        config = LLMConfig()
        assert config.provider.value == "ollama"
        assert config.model == "llama3.2"
        assert config.api_base == "http://localhost:11434"
        assert config.timeout == 60
    
    def test_custom_config(self):
        """Test custom LLM configuration."""
        from sentinxfl.llm.provider import LLMConfig, LLMProvider
        
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="mistral:7b",
            temperature=0.5,
            max_tokens=2048
        )
        assert config.model == "mistral:7b"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


class TestMockLLMProvider:
    """Test Mock LLM Provider."""
    
    @pytest.fixture
    def mock_provider(self):
        from sentinxfl.llm.provider import MockLLMProvider
        return MockLLMProvider()
    
    async def test_mock_generate(self, mock_provider):
        """Test mock text generation."""
        response = await mock_provider.generate("Test prompt")
        # MockLLMProvider returns LLMResponse object
        assert hasattr(response, 'content')
        assert len(response.content) > 0
    
    async def test_mock_chat(self, mock_provider):
        """Test mock chat."""
        from sentinxfl.llm.provider import ChatMessage
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = await mock_provider.chat(messages)
        # MockLLMProvider returns LLMResponse object
        assert hasattr(response, 'content')
        assert len(response.content) > 0
    
    async def test_mock_embed(self, mock_provider):
        """Test mock embeddings."""
        embeddings = await mock_provider.embed(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0  # Some embedding dimension
    
    async def test_mock_stream(self, mock_provider):
        """Test mock streaming."""
        chunks = []
        async for chunk in mock_provider.stream("Test prompt"):
            chunks.append(chunk)
        assert len(chunks) > 0
    
    async def test_mock_available(self, mock_provider):
        """Test mock availability check."""
        available = await mock_provider.is_available()
        assert available is True


class TestOllamaProvider:
    """Test Ollama LLM Provider."""
    
    async def test_ollama_initialization(self):
        """Test Ollama provider initialization."""
        from sentinxfl.llm.provider import OllamaProvider, LLMConfig
        
        config = LLMConfig(model="llama3.2:1b")
        provider = OllamaProvider(config)
        
        assert provider.config.model == "llama3.2:1b"
        assert provider.config.api_base == "http://localhost:11434"
    
    async def test_ollama_availability_check(self):
        """Test Ollama availability check (may fail if Ollama not running)."""
        from sentinxfl.llm.provider import OllamaProvider, LLMConfig
        
        config = LLMConfig()
        provider = OllamaProvider(config)
        
        # This just tests the method exists and runs
        # Actual availability depends on Ollama being installed
        available = await provider.is_available()
        assert isinstance(available, bool)


class TestLLMProviderFactory:
    """Test LLM provider factory function."""
    
    async def test_get_mock_provider(self):
        """Test getting mock provider."""
        from sentinxfl.llm.provider import get_llm_provider, LLMConfig, LLMProvider, MockLLMProvider
        
        config = LLMConfig(provider=LLMProvider.LOCAL)  # LOCAL falls back to mock internally
        provider = get_llm_provider(config)
        
        # Should be able to call methods
        response = await provider.generate("test")
        assert hasattr(response, 'content')
    
    async def test_get_ollama_provider(self):
        """Test getting Ollama provider."""
        from sentinxfl.llm.provider import get_llm_provider, LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.OLLAMA)
        provider = get_llm_provider(config)
        
        from sentinxfl.llm.provider import OllamaProvider
        assert isinstance(provider, OllamaProvider)


# ============================================================================
# RAG Pipeline Tests
# ============================================================================

class TestDocument:
    """Test Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a document."""
        from sentinxfl.llm.rag import Document
        
        doc = Document(
            id="test-001",
            content="This is test content",
            metadata={"type": "test"}
        )
        assert doc.id == "test-001"
        assert doc.content == "This is test content"
        assert doc.metadata["type"] == "test"


class TestRAGPipeline:
    """Test RAG Pipeline."""
    
    def test_document_class(self):
        """Test Document class creation."""
        from sentinxfl.llm.rag import Document
        
        doc = Document(
            id="test1",
            content="Test content",
            metadata={"type": "test"}
        )
        assert doc.id == "test1"
        assert doc.content == "Test content"
    
    def test_retrieval_result_class(self):
        """Test RetrievalResult class."""
        from sentinxfl.llm.rag import Document, RetrievalResult
        
        docs = [Document(id="1", content="test")]
        result = RetrievalResult(documents=docs, distances=[0.5], query="test query")
        
        assert len(result.documents) == 1
        assert result.query == "test query"


class TestFraudKnowledgeBase:
    """Test Fraud Knowledge Base."""
    
    def test_knowledge_base_has_patterns(self):
        """Test fraud knowledge base has fraud patterns."""
        from sentinxfl.llm.rag import FraudKnowledgeBase
        
        assert len(FraudKnowledgeBase.FRAUD_PATTERNS) > 0
        assert any("fraud" in p["content"].lower() for p in FraudKnowledgeBase.FRAUD_PATTERNS)
    
    def test_fraud_patterns_exist(self):
        """Test fraud patterns are in knowledge base."""
        from sentinxfl.llm.rag import FraudKnowledgeBase
        
        # Check for specific pattern types
        patterns = FraudKnowledgeBase.FRAUD_PATTERNS
        assert len(patterns) > 0
        assert all("id" in p and "content" in p for p in patterns)
    
    def test_detection_techniques_exist(self):
        """Test detection techniques are in knowledge base."""
        from sentinxfl.llm.rag import FraudKnowledgeBase
        
        techniques = FraudKnowledgeBase.DETECTION_TECHNIQUES
        assert len(techniques) > 0


# ============================================================================
# Fraud Explainer Tests
# ============================================================================

class TestFeatureContribution:
    """Test FeatureContribution dataclass."""
    
    def test_feature_contribution_creation(self):
        """Test creating a feature contribution."""
        from sentinxfl.llm.explainer import FeatureContribution
        
        fc = FeatureContribution(
            feature_name="amount",
            value=500.0,
            contribution=0.35,
            direction="increases",
            importance_rank=1
        )
        
        assert fc.feature_name == "amount"
        assert fc.value == 500.0
        assert fc.contribution == 0.35
        assert fc.direction == "increases"
    
    def test_feature_contribution_to_dict(self):
        """Test serialization."""
        from sentinxfl.llm.explainer import FeatureContribution
        
        fc = FeatureContribution(
            feature_name="velocity",
            value=5,
            contribution=0.2,
            direction="increases",
            importance_rank=2
        )
        
        d = fc.to_dict()
        assert d["feature_name"] == "velocity"
        assert d["value"] == 5


class TestExplanationType:
    """Test ExplanationType enum."""
    
    def test_explanation_types(self):
        """Test all explanation types exist."""
        from sentinxfl.llm.explainer import ExplanationType
        
        assert ExplanationType.BRIEF.value == "brief"
        assert ExplanationType.DETAILED.value == "detailed"
        assert ExplanationType.TECHNICAL.value == "technical"
        assert ExplanationType.EXECUTIVE.value == "executive"
        assert ExplanationType.REGULATORY.value == "regulatory"


class TestFraudExplainer:
    """Test Fraud Explainer."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        from sentinxfl.llm.provider import MockLLMProvider
        return MockLLMProvider()
    
    @pytest.fixture
    def explainer_config(self):
        """Create explainer config."""
        from sentinxfl.llm.explainer import ExplanationConfig
        return ExplanationConfig(
            top_k_features=3,
            include_rag_context=False,  # Disable RAG for unit tests
            include_recommendations=True
        )
    
    def test_risk_level_calculation(self):
        """Test risk level calculation."""
        from sentinxfl.llm.explainer import FraudExplainer
        
        explainer = FraudExplainer()
        
        # Thresholds: low<0.3<medium<0.5<...high<0.7<...critical>=0.9
        assert explainer._calculate_risk_level(0.1) == "low"
        assert explainer._calculate_risk_level(0.55) == "medium"  # >= 0.5
        assert explainer._calculate_risk_level(0.75) == "high"    # >= 0.7
        assert explainer._calculate_risk_level(0.95) == "critical"  # >= 0.9
    
    def test_feature_analysis(self):
        """Test feature analysis."""
        from sentinxfl.llm.explainer import FraudExplainer
        
        explainer = FraudExplainer()
        
        feature_values = {"amt": 500, "velocity_1h": 3, "distance_from_home": 50}
        feature_importances = {"amt": 0.3, "velocity_1h": 0.2, "distance_from_home": 0.1}
        
        contributions = explainer._analyze_features(
            feature_values, feature_importances, {}
        )
        
        assert len(contributions) == 3
        assert contributions[0].feature_name == "amt"  # Highest importance
        assert contributions[0].importance_rank == 1
    
    async def test_explain_high_risk(self, mock_llm_provider, explainer_config):
        """Test explanation generation for high-risk transaction."""
        from sentinxfl.llm.explainer import FraudExplainer
        
        explainer = FraudExplainer(
            llm_provider=mock_llm_provider,
            config=explainer_config
        )
        
        explanation = await explainer.explain(
            transaction_id="TXN_001",
            prediction=0.85,
            feature_values={
                "amt": 2500.0,
                "velocity_1h": 5,
                "distance_from_home": 200,
                "is_night": 1
            },
            feature_importances={
                "amt": 0.4,
                "velocity_1h": 0.3,
                "distance_from_home": 0.2,
                "is_night": 0.1
            },
            model_name="test_model"
        )
        
        assert explanation.transaction_id == "TXN_001"
        assert explanation.prediction == 0.85
        assert explanation.is_fraud is True
        assert explanation.risk_level == "high"
        assert len(explanation.top_features) <= 3
        assert len(explanation.recommended_actions) > 0
    
    async def test_explain_low_risk(self, mock_llm_provider, explainer_config):
        """Test explanation generation for low-risk transaction."""
        from sentinxfl.llm.explainer import FraudExplainer
        
        explainer = FraudExplainer(
            llm_provider=mock_llm_provider,
            config=explainer_config
        )
        
        explanation = await explainer.explain(
            transaction_id="TXN_002",
            prediction=0.15,
            feature_values={
                "amt": 50.0,
                "velocity_1h": 1,
                "distance_from_home": 5
            },
            model_name="test_model"
        )
        
        assert explanation.is_fraud is False
        assert explanation.risk_level == "low"
    
    def test_pattern_detection_heuristic(self, mock_llm_provider, explainer_config):
        """Test heuristic pattern detection."""
        from sentinxfl.llm.explainer import FraudExplainer
        
        explainer = FraudExplainer(
            llm_provider=mock_llm_provider,
            config=explainer_config
        )
        
        patterns, confidence = explainer._detect_patterns_heuristic(
            feature_values={
                "amount_zscore": 3.5,
                "velocity_1h": 5,
                "distance_from_home": 150,
                "is_night": 1
            },
            prediction=0.8
        )
        
        assert len(patterns) > 0
        assert confidence > 0.5
        assert any("amount" in p.lower() for p in patterns)
    
    async def test_batch_explain(self, mock_llm_provider, explainer_config):
        """Test batch explanation generation."""
        from sentinxfl.llm.explainer import FraudExplainer
        
        explainer = FraudExplainer(
            llm_provider=mock_llm_provider,
            config=explainer_config
        )
        
        predictions = [
            {"transaction_id": "TXN_001", "prediction": 0.9, "feature_values": {"amt": 1000}},
            {"transaction_id": "TXN_002", "prediction": 0.3, "feature_values": {"amt": 50}},
        ]
        
        explanations = await explainer.batch_explain(predictions, batch_size=2)
        
        assert len(explanations) == 2
        assert explanations[0].transaction_id == "TXN_001"
        assert explanations[1].transaction_id == "TXN_002"


class TestFraudExplanation:
    """Test FraudExplanation dataclass."""
    
    def test_explanation_to_dict(self):
        """Test explanation serialization."""
        from sentinxfl.llm.explainer import (
            FraudExplanation,
            FeatureContribution,
            ExplanationType
        )
        
        explanation = FraudExplanation(
            transaction_id="TXN_001",
            prediction=0.8,
            is_fraud=True,
            risk_level="high",
            top_features=[
                FeatureContribution("amt", 500, 0.3, "increases", 1)
            ],
            feature_summary="amount increases risk",
            detected_patterns=["High amount"],
            pattern_confidence=0.7,
            brief_explanation="High risk due to amount",
            detailed_explanation="Detailed analysis...",
            recommended_actions=["Review transaction"],
            model_name="xgboost",
            explanation_type=ExplanationType.DETAILED,
            confidence_score=0.85
        )
        
        d = explanation.to_dict()
        
        assert d["transaction_id"] == "TXN_001"
        assert d["prediction"] == 0.8
        assert d["is_fraud"] is True
        assert len(d["top_features"]) == 1


# ============================================================================
# LLM API Routes Tests
# ============================================================================

class TestLLMAPIRoutes:
    """Test LLM API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from sentinxfl.api.app import create_app
        
        app = create_app()
        return TestClient(app)
    
    def test_explain_endpoint_exists(self, client):
        """Test explain endpoint is registered."""
        # POST without body should return 422 (validation error), not 404
        response = client.post("/api/v1/llm/explain")
        assert response.status_code != 404
    
    def test_rag_query_endpoint_exists(self, client):
        """Test RAG query endpoint is registered."""
        response = client.post("/api/v1/llm/rag/query")
        assert response.status_code != 404
    
    def test_chat_endpoint_exists(self, client):
        """Test chat endpoint is registered."""
        response = client.post("/api/v1/llm/chat")
        assert response.status_code != 404
    
    def test_health_endpoint(self, client):
        """Test LLM health endpoint."""
        response = client.get("/api/v1/llm/health")
        # Should return 200 even if services are unavailable
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "llm_available" in data
        assert "rag_initialized" in data


# ============================================================================
# Integration Tests
# ============================================================================

class TestLLMIntegration:
    """Integration tests requiring all components."""
    
    async def test_full_explanation_pipeline(self):
        """Test full explanation pipeline without external LLM."""
        from sentinxfl.llm.explainer import (
            FraudExplainer,
            ExplanationConfig,
        )
        from sentinxfl.llm.provider import MockLLMProvider
        
        # Create explainer with mock LLM (no external dependencies)
        config = ExplanationConfig(
            include_rag_context=False,
            top_k_features=5
        )
        explainer = FraudExplainer(
            llm_provider=MockLLMProvider(),
            config=config
        )
        
        # Generate explanation
        explanation = await explainer.explain(
            transaction_id="INT_TEST_001",
            prediction=0.72,
            feature_values={
                "amt": 1500.0,
                "amount_zscore": 2.1,
                "velocity_1h": 3,
                "distance_from_home": 100,
                "trans_hour": 3,
                "is_night": 1,
                "merchant_fraud_rate": 0.08
            },
            model_name="integration_test"
        )
        
        # Verify all components work together
        assert explanation.transaction_id == "INT_TEST_001"
        assert explanation.risk_level == "high"
        assert explanation.is_fraud is True
        assert len(explanation.brief_explanation) > 0
        assert len(explanation.detailed_explanation) > 0
        assert len(explanation.detected_patterns) > 0
        assert len(explanation.top_features) > 0
        assert 0 <= explanation.confidence_score <= 1
    
    def test_fraud_knowledge_base_structure(self):
        """Test FraudKnowledgeBase structure for loading."""
        from sentinxfl.llm.rag import FraudKnowledgeBase
        
        # Check all required data exists
        assert len(FraudKnowledgeBase.FRAUD_PATTERNS) > 0
        assert len(FraudKnowledgeBase.DETECTION_TECHNIQUES) > 0
        assert len(FraudKnowledgeBase.INTERPRETABILITY) > 0
        
        # Check structure
        for pattern in FraudKnowledgeBase.FRAUD_PATTERNS:
            assert "id" in pattern
            assert "content" in pattern
            assert "metadata" in pattern


# ============================================================================
# Module Import Tests
# ============================================================================

class TestModuleImports:
    """Test that all LLM modules import correctly."""
    
    def test_import_provider(self):
        """Test provider module imports."""
        from sentinxfl.llm.provider import (
            BaseLLMProvider,
            OllamaProvider,
            MockLLMProvider,
            LLMConfig,
            ChatMessage,
            get_llm_provider
        )
        assert BaseLLMProvider is not None
        assert OllamaProvider is not None
        assert MockLLMProvider is not None
    
    def test_import_rag(self):
        """Test RAG module imports."""
        from sentinxfl.llm.rag import (
            RAGPipeline,
            Document,
            RetrievalResult,
            RAGResponse,
            FraudKnowledgeBase
        )
        assert RAGPipeline is not None
        assert Document is not None
    
    def test_import_explainer(self):
        """Test explainer module imports."""
        from sentinxfl.llm.explainer import (
            FraudExplainer,
            FraudExplanation,
            FeatureContribution,
            ExplanationType,
            ExplanationConfig,
            create_explainer
        )
        assert FraudExplainer is not None
        assert ExplanationType is not None
    
    def test_import_from_init(self):
        """Test imports from __init__.py."""
        from sentinxfl.llm import (
            BaseLLMProvider,
            OllamaProvider,
            RAGPipeline,
            FraudExplainer,
            ExplanationType
        )
        assert all([
            BaseLLMProvider,
            OllamaProvider,
            RAGPipeline,
            FraudExplainer,
            ExplanationType
        ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
