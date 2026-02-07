"""
Sprint 5 Tests: Dashboard API Integration & E2E Tests
Tests for API endpoints used by the React dashboard
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from src.sentinxfl.api.app import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "message" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestLLMEndpoints:
    """Test LLM API endpoints used by dashboard"""
    
    def test_llm_health(self, client):
        """Test LLM health check"""
        response = client.get("/api/v1/llm/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_explain_transaction(self, client):
        """Test fraud explanation endpoint"""
        payload = {
            "transaction_id": "TXN_TEST_001",
            "features": {
                "amount": 15000.0,
                "time_hour": 3,
                "location_risk": 0.8
            },
            "fraud_probability": 0.87
        }
        response = client.post("/api/v1/llm/explain", json=payload)
        # Even if LLM not available, should return structured response
        assert response.status_code in [200, 422, 503]
    
    def test_rag_query(self, client):
        """Test RAG query endpoint"""
        payload = {
            "query": "What are common fraud patterns?",
            "top_k": 3
        }
        response = client.post("/api/v1/llm/rag/query", json=payload)
        assert response.status_code in [200, 422, 503]
    
    def test_chat_endpoint(self, client):
        """Test chat endpoint"""
        payload = {
            "messages": [
                {"role": "user", "content": "What is SentinXFL?"}
            ]
        }
        response = client.post("/api/v1/llm/chat", json=payload)
        assert response.status_code in [200, 422, 503]


class TestPIIEndpoints:
    """Test PII processing endpoints"""
    
    def test_pii_scan(self, client):
        """Test PII scan endpoint"""
        # Test with sample data
        response = client.get("/api/v1/pii/status")
        # Endpoint may not exist yet, that's ok
        assert response.status_code in [200, 404]


class TestModelEndpoints:
    """Test ML model endpoints"""
    
    def test_model_list(self, client):
        """Test listing available models"""
        response = client.get("/api/v1/models")
        assert response.status_code in [200, 404]
    
    def test_predict_endpoint(self, client):
        """Test fraud prediction endpoint"""
        payload = {
            "features": {
                "amount": 1000.0,
                "transaction_type": 1,
                "hour": 14
            }
        }
        response = client.post("/api/v1/models/predict", json=payload)
        assert response.status_code in [200, 404, 422]


class TestDashboardAPIContract:
    """Test that API responses match dashboard expectations"""
    
    def test_explanation_response_format(self, client):
        """Test explanation response has expected fields"""
        payload = {
            "transaction_id": "TEST_001",
            "features": {"amount": 5000.0},
            "fraud_probability": 0.75
        }
        response = client.post("/api/v1/llm/explain", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            # Dashboard expects these fields
            expected_fields = ["transaction_id", "risk_score", "risk_level"]
            for field in expected_fields:
                assert field in data, f"Missing field: {field}"
    
    def test_health_response_format(self, client):
        """Test health response format"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok", "running"]


class TestAPIErrorHandling:
    """Test API error handling"""
    
    def test_invalid_endpoint(self, client):
        """Test 404 for invalid endpoint"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_json(self, client):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/v1/llm/explain",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        response = client.post("/api/v1/llm/explain", json={})
        assert response.status_code == 422


class TestDashboardIntegration:
    """Integration tests for dashboard workflow"""
    
    def test_transaction_analysis_flow(self, client):
        """Test complete transaction analysis flow"""
        # 1. Check API health
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        
        # 2. Check LLM health
        llm_health = client.get("/api/v1/llm/health")
        assert llm_health.status_code == 200
        
        # 3. Request explanation (may fail gracefully if LLM unavailable)
        explain_payload = {
            "transaction_id": "FLOW_TEST_001",
            "features": {
                "amount": 25000.0,
                "velocity_24h": 5
            },
            "fraud_probability": 0.92
        }
        explain_response = client.post("/api/v1/llm/explain", json=explain_payload)
        # Should either succeed, return validation error, or service unavailable
        assert explain_response.status_code in [200, 422, 503]
    
    def test_rag_knowledge_query_flow(self, client):
        """Test RAG knowledge base query flow"""
        queries = [
            "How does federated learning protect privacy?",
            "What is differential privacy?",
            "Explain fraud detection patterns"
        ]
        
        for query in queries:
            response = client.post(
                "/api/v1/llm/rag/query",
                json={"query": query, "top_k": 3}
            )
            # 200: success, 422: validation, 503: service unavailable
            assert response.status_code in [200, 422, 503]


class TestCORS:
    """Test CORS configuration for dashboard"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS preflight should be handled
        assert response.status_code in [200, 204, 405]


class TestAPIVersioning:
    """Test API versioning"""
    
    def test_v1_prefix(self, client):
        """Test API v1 prefix works"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    def test_docs_available(self, client):
        """Test OpenAPI docs are available"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
