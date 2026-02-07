"""
SentinXFL - RAG Pipeline
=========================

Retrieval-Augmented Generation pipeline using ChromaDB
for context-aware fraud explanations.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.llm.provider import ChatMessage, get_llm_provider

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class Document:
    """Document for RAG storage."""
    
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    
    documents: list[Document]
    distances: list[float]
    query: str


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""
    
    answer: str
    sources: list[Document]
    context_used: str
    model: str
    tokens_used: int = 0


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    Uses ChromaDB for vector storage and retrieval,
    with LLM for generation.
    """
    
    def __init__(
        self,
        collection_name: str = "sentinxfl_knowledge",
        persist_directory: str | None = None,
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            collection_name: ChromaDB collection name
            persist_directory: Optional persistence path
        """
        self.collection_name = collection_name
        
        # ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        # LLM provider
        self.llm = get_llm_provider()
        
        logger.info(f"RAG Pipeline initialized: collection={collection_name}")
    
    async def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: Documents to add
            batch_size: Batch size for embedding
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = await self.llm.embed(texts)
        
        # Add to ChromaDB
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
        
        # Batch insert
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            self.collection.add(
                ids=[doc.id for doc in batch],
                embeddings=[doc.embedding for doc in batch],
                documents=[doc.content for doc in batch],
                metadatas=[doc.metadata for doc in batch],
            )
        
        logger.info(f"Added {len(documents)} documents to RAG")
        return len(documents)
    
    async def add_text(
        self,
        text: str,
        doc_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """
        Add a single text to the knowledge base.
        
        Args:
            text: Text content
            doc_id: Optional document ID
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        import uuid
        
        doc_id = doc_id or str(uuid.uuid4())
        
        doc = Document(
            id=doc_id,
            content=text,
            metadata=metadata or {},
        )
        
        await self.add_documents([doc])
        
        return doc_id
    
    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            n_results: Number of results
            where: Optional ChromaDB filter
            
        Returns:
            Retrieval results
        """
        # Get query embedding
        embeddings = await self.llm.embed([query])
        query_embedding = embeddings[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Parse results
        documents = []
        distances = []
        
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                doc = Document(
                    id=doc_id,
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                )
                documents.append(doc)
                distances.append(results["distances"][0][i] if results["distances"] else 0.0)
        
        return RetrievalResult(
            documents=documents,
            distances=distances,
            query=query,
        )
    
    async def query(
        self,
        question: str,
        n_context: int = 5,
        system_prompt: str | None = None,
    ) -> RAGResponse:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            n_context: Number of context documents
            system_prompt: Optional custom system prompt
            
        Returns:
            RAG response with answer and sources
        """
        # Retrieve relevant documents
        retrieval = await self.retrieve(question, n_results=n_context)
        
        # Build context
        context_parts = []
        for i, doc in enumerate(retrieval.documents):
            context_parts.append(f"[Source {i+1}]: {doc.content}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        rag_prompt = f"""Based on the following context, answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate response
        response = await self.llm.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt or """You are a fraud detection expert assistant.
Answer questions based on the provided context. Be accurate and concise.
If the context doesn't contain the answer, acknowledge that.""",
        )
        
        return RAGResponse(
            answer=response.content,
            sources=retrieval.documents,
            context_used=context,
            model=response.model,
            tokens_used=response.tokens_used,
        )
    
    async def chat_with_context(
        self,
        messages: list[ChatMessage],
        n_context: int = 3,
    ) -> RAGResponse:
        """
        Chat with RAG context injection.
        
        Args:
            messages: Conversation messages
            n_context: Number of context docs per turn
            
        Returns:
            RAG response
        """
        # Get last user message for retrieval
        user_messages = [m for m in messages if m.role == "user"]
        if not user_messages:
            return RAGResponse(
                answer="Please ask a question.",
                sources=[],
                context_used="",
                model=self.llm.config.model,
            )
        
        last_question = user_messages[-1].content
        
        # Retrieve context
        retrieval = await self.retrieve(last_question, n_results=n_context)
        
        # Build context-augmented prompt
        context_parts = [doc.content for doc in retrieval.documents]
        context = "\n\n".join(context_parts)
        
        # Insert context into system message
        system_msg = ChatMessage(
            role="system",
            content=f"""You are a fraud detection expert assistant for SentinXFL.
Use the following context to help answer questions:

{context}

Be accurate, concise, and helpful.""",
        )
        
        # Build message list
        augmented_messages = [system_msg] + [m for m in messages if m.role != "system"]
        
        # Generate
        response = await self.llm.chat(augmented_messages)
        
        return RAGResponse(
            answer=response.content,
            sources=retrieval.documents,
            context_used=context,
            model=response.model,
            tokens_used=response.tokens_used,
        )
    
    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
        }
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")


# ==========================================
# Knowledge Base Loader
# ==========================================

class FraudKnowledgeBase:
    """
    Pre-built knowledge base for fraud detection.
    
    Contains domain knowledge about:
    - Fraud patterns
    - Detection techniques
    - Feature engineering
    - Model interpretation
    """
    
    FRAUD_PATTERNS = [
        {
            "id": "pattern_card_testing",
            "content": """Card Testing Fraud: Criminals test stolen cards with small transactions 
to verify validity. Indicators: Multiple small transactions (<$5) in quick succession, 
different merchants, often online. Detection: Monitor velocity of low-value transactions, 
check for sequential card numbers, time between transactions.""",
            "metadata": {"category": "fraud_pattern", "severity": "high"},
        },
        {
            "id": "pattern_account_takeover",
            "content": """Account Takeover (ATO): Criminals gain access to legitimate accounts 
through phishing, credential stuffing, or social engineering. Indicators: Login from new device/IP, 
password change followed by high-value transactions, shipping address change. 
Detection: Device fingerprinting, behavioral biometrics, anomaly detection.""",
            "metadata": {"category": "fraud_pattern", "severity": "critical"},
        },
        {
            "id": "pattern_friendly_fraud",
            "content": """Friendly Fraud (Chargeback Abuse): Legitimate cardholders dispute valid 
transactions to get refunds while keeping goods. Indicators: Repeated chargebacks from same 
customer, pattern of disputes after delivery confirmation. Detection: Track chargeback history, 
delivery signature verification, customer dispute patterns.""",
            "metadata": {"category": "fraud_pattern", "severity": "medium"},
        },
        {
            "id": "pattern_bust_out",
            "content": """Bust-Out Fraud: Building credit history then maxing out accounts with 
no intent to repay. Indicators: Sudden increase in credit utilization, multiple new accounts, 
cash advances, quick balance transfers. Detection: Credit velocity monitoring, 
utilization change alerts.""",
            "metadata": {"category": "fraud_pattern", "severity": "high"},
        },
        {
            "id": "pattern_synthetic_identity",
            "content": """Synthetic Identity Fraud: Combining real and fake information to 
create new identities. Uses real SSN (often from children/elderly) with fake name/DOB. 
Indicators: SSN issued recently but applicant age doesn't match, thin credit file, 
PO Box addresses. Detection: SSN validation, identity verification services.""",
            "metadata": {"category": "fraud_pattern", "severity": "critical"},
        },
    ]
    
    DETECTION_TECHNIQUES = [
        {
            "id": "tech_anomaly_detection",
            "content": """Anomaly Detection for Fraud: Unsupervised learning to identify 
unusual transactions. Methods: Isolation Forest (tree-based isolation), 
One-Class SVM (boundary learning), Autoencoders (reconstruction error). 
Best for: Zero-day fraud, emerging patterns. Limitations: High false positive rate.""",
            "metadata": {"category": "technique", "type": "ml"},
        },
        {
            "id": "tech_gradient_boosting",
            "content": """Gradient Boosting (XGBoost/LightGBM) for Fraud: Supervised learning 
combining weak learners. Advantages: Handles imbalanced data, feature importance, 
fast inference. Parameters: scale_pos_weight for class imbalance, max_depth for complexity. 
Best for: Known fraud patterns with labeled data.""",
            "metadata": {"category": "technique", "type": "ml"},
        },
        {
            "id": "tech_feature_engineering",
            "content": """Fraud Detection Feature Engineering: Key features include 
transaction velocity (count/sum in time windows), merchant category risk scores, 
time-based patterns (hour, day, weekend), distance from home/usual locations, 
device fingerprints, behavioral features (typing speed, mouse patterns). 
Aggregation windows: 1hr, 24hr, 7days, 30days.""",
            "metadata": {"category": "technique", "type": "feature"},
        },
        {
            "id": "tech_federated_learning",
            "content": """Federated Learning for Fraud Detection: Train models across 
multiple banks without sharing raw data. Benefits: Privacy preservation, 
regulatory compliance (GDPR, CCPA), larger effective dataset. Challenges: 
Non-IID data distribution, communication overhead, Byzantine resilience. 
Aggregation methods: FedAvg, Multi-Krum, Trimmed Mean.""",
            "metadata": {"category": "technique", "type": "fl"},
        },
    ]
    
    INTERPRETABILITY = [
        {
            "id": "interp_shap",
            "content": """SHAP (SHapley Additive exPlanations) for Fraud Models: 
Game-theoretic approach to explain predictions. Provides: Feature importance per prediction, 
global feature ranking, interaction effects. Visualization: Force plots, summary plots. 
Use for: Regulatory compliance, analyst review, model debugging.""",
            "metadata": {"category": "interpretability", "tool": "shap"},
        },
        {
            "id": "interp_tabnet",
            "content": """TabNet Attention for Fraud: Neural network with built-in 
feature selection via attention mechanism. Provides: Per-prediction feature masks, 
sequential decision steps, interpretable without post-hoc methods. 
Use for: End-to-end interpretable deep learning on tabular fraud data.""",
            "metadata": {"category": "interpretability", "tool": "tabnet"},
        },
    ]
    
    @classmethod
    async def load_to_pipeline(cls, pipeline: RAGPipeline) -> int:
        """Load fraud knowledge into RAG pipeline."""
        documents = []
        
        for item in cls.FRAUD_PATTERNS:
            documents.append(Document(
                id=item["id"],
                content=item["content"],
                metadata=item["metadata"],
            ))
        
        for item in cls.DETECTION_TECHNIQUES:
            documents.append(Document(
                id=item["id"],
                content=item["content"],
                metadata=item["metadata"],
            ))
        
        for item in cls.INTERPRETABILITY:
            documents.append(Document(
                id=item["id"],
                content=item["content"],
                metadata=item["metadata"],
            ))
        
        return await pipeline.add_documents(documents)


# ==========================================
# Global Pipeline Instance
# ==========================================

_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline(
    collection_name: str = "sentinxfl_knowledge",
    persist_directory: str | None = None,
) -> RAGPipeline:
    """Get or create RAG pipeline instance."""
    global _rag_pipeline
    
    if _rag_pipeline is None:
        persist_dir = persist_directory or str(
            Path(settings.data_dir) / "chroma"
        )
        _rag_pipeline = RAGPipeline(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
    
    return _rag_pipeline
