"""
LLM API Routes for SentinXFL.

Provides REST endpoints for:
- Fraud prediction explanations
- RAG-based question answering
- Chat interface for fraud analysis
"""

from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from loguru import logger

from sentinxfl.llm import (
    FraudExplainer,
    ExplanationType,
    ExplanationConfig,
    RAGPipeline,
    FraudKnowledgeBase,
    get_llm_provider,
    ChatMessage,
)


router = APIRouter(prefix="/llm", tags=["LLM & Explainability"])

# Global instances (initialized on startup)
_explainer: FraudExplainer | None = None
_rag_pipeline: RAGPipeline | None = None


# ============================================================================
# Request/Response Models
# ============================================================================

class ExplainRequest(BaseModel):
    """Request for fraud prediction explanation."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    prediction: float = Field(..., ge=0, le=1, description="Fraud probability (0-1)")
    feature_values: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of feature name -> value"
    )
    feature_importances: dict[str, float] | None = Field(
        None, description="Optional feature importance scores"
    )
    shap_values: dict[str, float] | None = Field(
        None, description="Optional SHAP values"
    )
    model_name: str = Field("ensemble", description="Model that made the prediction")
    explanation_type: str = Field(
        "detailed",
        description="Type: brief, detailed, technical, executive, regulatory"
    )


class ExplainResponse(BaseModel):
    """Response with fraud explanation."""
    transaction_id: str
    prediction: float
    is_fraud: bool
    risk_level: str
    brief_explanation: str
    detailed_explanation: str
    top_features: list[dict[str, Any]]
    detected_patterns: list[str]
    recommendations: list[str]
    confidence_score: float


class BatchExplainRequest(BaseModel):
    """Request for batch explanations."""
    predictions: list[dict[str, Any]] = Field(
        ..., description="List of prediction dicts with transaction_id, prediction, feature_values"
    )
    batch_size: int = Field(10, ge=1, le=50, description="Concurrent batch size")


class RAGQueryRequest(BaseModel):
    """Request for RAG-based question."""
    question: str = Field(..., min_length=3, max_length=1000, description="Question to ask")
    top_k: int = Field(3, ge=1, le=10, description="Number of documents to retrieve")
    include_context: bool = Field(True, description="Include retrieved context in response")


class RAGQueryResponse(BaseModel):
    """Response from RAG query."""
    answer: str
    sources: list[dict[str, Any]]
    confidence: float


class ChatRequest(BaseModel):
    """Request for chat interaction."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Chat history as list of {role, content} dicts"
    )
    use_rag: bool = Field(True, description="Whether to use RAG for context")


class ChatResponse(BaseModel):
    """Response from chat."""
    response: str
    sources: list[str] = Field(default_factory=list)


class AddDocumentRequest(BaseModel):
    """Request to add document to knowledge base."""
    content: str = Field(..., min_length=10, max_length=10000, description="Document content")
    doc_id: str | None = Field(None, description="Optional document ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class LLMHealthResponse(BaseModel):
    """Health check response."""
    status: str
    llm_available: bool
    rag_initialized: bool
    document_count: int
    model_name: str


# ============================================================================
# Helper Functions
# ============================================================================

async def get_explainer() -> FraudExplainer:
    """Get or create the fraud explainer instance."""
    global _explainer
    if _explainer is None:
        _explainer = FraudExplainer()
        await _explainer.initialize()
    return _explainer


async def get_rag_pipeline() -> RAGPipeline:
    """Get or create the RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
        await _rag_pipeline.initialize()
        # Load fraud knowledge base
        knowledge_base = FraudKnowledgeBase()
        await knowledge_base.load_to_pipeline(_rag_pipeline)
        logger.info("RAG pipeline initialized with fraud knowledge base")
    return _rag_pipeline


# ============================================================================
# Explanation Endpoints
# ============================================================================

@router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(request: ExplainRequest) -> ExplainResponse:
    """
    Generate a human-readable explanation for a fraud prediction.
    
    Uses feature importance analysis, pattern detection, and LLM
    to create comprehensive explanations.
    """
    try:
        explainer = await get_explainer()
        
        # Map string to enum
        exp_type = ExplanationType(request.explanation_type)
        
        explanation = await explainer.explain(
            transaction_id=request.transaction_id,
            prediction=request.prediction,
            feature_values=request.feature_values,
            feature_importances=request.feature_importances,
            shap_values=request.shap_values,
            model_name=request.model_name,
            explanation_type=exp_type
        )
        
        return ExplainResponse(
            transaction_id=explanation.transaction_id,
            prediction=explanation.prediction,
            is_fraud=explanation.is_fraud,
            risk_level=explanation.risk_level,
            brief_explanation=explanation.brief_explanation,
            detailed_explanation=explanation.detailed_explanation,
            top_features=[f.to_dict() for f in explanation.top_features],
            detected_patterns=explanation.detected_patterns,
            recommendations=explanation.recommended_actions,
            confidence_score=explanation.confidence_score
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


@router.post("/explain/batch")
async def batch_explain_predictions(
    request: BatchExplainRequest,
    background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """
    Generate explanations for multiple predictions in batch.
    
    Returns explanations for all provided predictions.
    For large batches (>50), consider using async processing.
    """
    try:
        explainer = await get_explainer()
        
        explanations = await explainer.batch_explain(
            predictions=request.predictions,
            batch_size=request.batch_size
        )
        
        return {
            "total": len(request.predictions),
            "successful": len(explanations),
            "explanations": [exp.to_dict() for exp in explanations]
        }
        
    except Exception as e:
        logger.error(f"Batch explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RAG Endpoints
# ============================================================================

@router.post("/rag/query", response_model=RAGQueryResponse)
async def query_knowledge_base(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Query the fraud knowledge base using RAG.
    
    Retrieves relevant documents and generates an answer
    using the LLM with retrieved context.
    """
    try:
        rag = await get_rag_pipeline()
        
        response = await rag.query(
            question=request.question,
            top_k=request.top_k
        )
        
        sources = []
        if request.include_context:
            for result in response.sources:
                sources.append({
                    "content": result.document.content[:500],
                    "score": result.score,
                    "metadata": result.document.metadata
                })
        
        return RAGQueryResponse(
            answer=response.answer,
            sources=sources,
            confidence=response.confidence
        )
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/documents")
async def add_document(request: AddDocumentRequest) -> dict[str, str]:
    """
    Add a document to the fraud knowledge base.
    
    Documents are stored in ChromaDB and used for RAG queries.
    """
    try:
        rag = await get_rag_pipeline()
        
        from sentinxfl.llm.rag import Document
        
        doc = Document(
            id=request.doc_id or f"doc_{hash(request.content)}",
            content=request.content,
            metadata=request.metadata
        )
        
        await rag.add_documents([doc])
        
        return {"status": "success", "document_id": doc.id}
        
    except Exception as e:
        logger.error(f"Document add failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/documents/count")
async def get_document_count() -> dict[str, int]:
    """Get the number of documents in the knowledge base."""
    try:
        rag = await get_rag_pipeline()
        count = await rag.get_document_count()
        return {"count": count}
    except Exception as e:
        logger.error(f"Document count failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat interface for fraud analysis questions.
    
    Uses conversation history and optionally RAG context
    to provide informed responses about fraud detection.
    """
    try:
        llm = await get_llm_provider()
        sources = []
        
        # Build context from RAG if enabled
        context = ""
        if request.use_rag:
            try:
                rag = await get_rag_pipeline()
                results = await rag.retrieve(request.message, top_k=3)
                context = "\n\n".join([
                    f"Reference: {r.document.content[:300]}"
                    for r in results
                ])
                sources = [r.document.metadata.get("title", r.document.id) for r in results]
            except Exception as e:
                logger.warning(f"RAG retrieval failed in chat: {e}")
        
        # Build message history
        messages = [
            ChatMessage(
                role="system",
                content="""You are a fraud detection expert assistant for SentinXFL, 
a privacy-preserving federated learning platform for financial fraud detection.
Help users understand fraud patterns, model predictions, and security best practices.
Be concise but informative. If you reference specific knowledge, cite it."""
            )
        ]
        
        # Add context if available
        if context:
            messages.append(ChatMessage(
                role="system",
                content=f"Relevant knowledge base context:\n{context}"
            ))
        
        # Add chat history
        for msg in request.history[-10:]:  # Limit history to last 10
            messages.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", "")
            ))
        
        # Add current message
        messages.append(ChatMessage(role="user", content=request.message))
        
        # Generate response
        response = await llm.chat(messages, temperature=0.7, max_tokens=1024)
        
        return ChatResponse(
            response=response,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Management Endpoints
# ============================================================================

@router.get("/health", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """
    Check the health of LLM and RAG systems.
    """
    try:
        # Check LLM
        llm_available = False
        model_name = "unknown"
        try:
            llm = await get_llm_provider()
            llm_available = await llm.is_available()
            model_name = llm.model_name
        except Exception:
            pass
        
        # Check RAG
        rag_initialized = False
        doc_count = 0
        try:
            rag = await get_rag_pipeline()
            rag_initialized = rag._initialized
            doc_count = await rag.get_document_count()
        except Exception:
            pass
        
        status = "healthy" if llm_available and rag_initialized else "degraded"
        
        return LLMHealthResponse(
            status=status,
            llm_available=llm_available,
            rag_initialized=rag_initialized,
            document_count=doc_count,
            model_name=model_name
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return LLMHealthResponse(
            status="unhealthy",
            llm_available=False,
            rag_initialized=False,
            document_count=0,
            model_name="error"
        )


@router.post("/initialize")
async def initialize_llm_services() -> dict[str, str]:
    """
    Initialize LLM and RAG services.
    
    Call this endpoint to pre-warm the services before use.
    """
    try:
        await get_explainer()
        await get_rag_pipeline()
        return {"status": "initialized", "message": "LLM and RAG services are ready"}
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/reset")
async def reset_knowledge_base() -> dict[str, str]:
    """
    Reset the knowledge base (delete all documents and reinitialize).
    
    WARNING: This will delete all custom documents. Fraud knowledge base
    will be reloaded automatically.
    """
    global _rag_pipeline
    try:
        _rag_pipeline = None
        await get_rag_pipeline()  # Reinitialize with base knowledge
        return {"status": "success", "message": "Knowledge base reset and reinitialized"}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
