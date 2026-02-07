"""
Fraud Explanation Generator for SentinXFL.

Generates human-readable explanations for fraud predictions using:
- Feature importance analysis
- SHAP values interpretation
- RAG-enhanced context
- LLM-based natural language generation
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from sentinxfl.llm.provider import BaseLLMProvider, get_llm_provider, LLMConfig
from sentinxfl.llm.rag import RAGPipeline, FraudKnowledgeBase


class ExplanationType(str, Enum):
    """Types of explanations."""
    BRIEF = "brief"  # One-line summary
    DETAILED = "detailed"  # Comprehensive analysis
    TECHNICAL = "technical"  # For data scientists
    EXECUTIVE = "executive"  # For business stakeholders
    REGULATORY = "regulatory"  # For compliance reporting


@dataclass
class FeatureContribution:
    """A feature's contribution to the prediction."""
    feature_name: str
    value: float  # The actual feature value
    contribution: float  # SHAP or importance score
    direction: str  # "increases" or "decreases" fraud risk
    importance_rank: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "contribution": self.contribution,
            "direction": self.direction,
            "importance_rank": self.importance_rank
        }


@dataclass
class FraudExplanation:
    """Complete explanation for a fraud prediction."""
    transaction_id: str
    prediction: float  # Probability of fraud (0-1)
    is_fraud: bool  # Binary decision
    risk_level: str  # "low", "medium", "high", "critical"
    
    # Feature analysis
    top_features: list[FeatureContribution]
    feature_summary: str
    
    # Pattern analysis
    detected_patterns: list[str]
    pattern_confidence: float
    
    # Natural language explanation
    brief_explanation: str
    detailed_explanation: str
    
    # Recommendations
    recommended_actions: list[str]
    
    # Metadata
    model_name: str
    explanation_type: ExplanationType
    confidence_score: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "prediction": self.prediction,
            "is_fraud": self.is_fraud,
            "risk_level": self.risk_level,
            "top_features": [f.to_dict() for f in self.top_features],
            "feature_summary": self.feature_summary,
            "detected_patterns": self.detected_patterns,
            "pattern_confidence": self.pattern_confidence,
            "brief_explanation": self.brief_explanation,
            "detailed_explanation": self.detailed_explanation,
            "recommended_actions": self.recommended_actions,
            "model_name": self.model_name,
            "explanation_type": self.explanation_type.value,
            "confidence_score": self.confidence_score
        }


@dataclass
class ExplanationConfig:
    """Configuration for explanation generation."""
    top_k_features: int = 5
    include_rag_context: bool = True
    rag_top_k: int = 3
    explanation_type: ExplanationType = ExplanationType.DETAILED
    temperature: float = 0.3  # Lower for more consistent explanations
    max_tokens: int = 1024
    include_recommendations: bool = True


class FraudExplainer:
    """
    Generates human-readable explanations for fraud predictions.
    
    Uses a combination of:
    - Feature importance/SHAP values for understanding what drove the prediction
    - RAG pipeline for domain context about fraud patterns
    - LLM for natural language generation
    """
    
    # Feature name mappings for human-readable descriptions
    FEATURE_DESCRIPTIONS: dict[str, str] = {
        "amt": "transaction amount",
        "trans_hour": "hour of transaction",
        "trans_day_of_week": "day of week",
        "age": "customer age",
        "city_pop": "city population",
        "merchant_fraud_rate": "merchant's historical fraud rate",
        "category_fraud_rate": "category fraud rate",
        "distance_from_home": "distance from home address",
        "time_since_last_trans": "time since last transaction",
        "velocity_1h": "transaction velocity (1 hour)",
        "velocity_24h": "transaction velocity (24 hours)",
        "amount_zscore": "amount deviation from normal",
        "is_weekend": "weekend indicator",
        "is_night": "nighttime indicator",
        "merchant_category": "merchant category",
        "is_online": "online transaction",
        "transaction_frequency": "transaction frequency",
        "avg_transaction_amount": "average spending amount",
        "max_transaction_amount": "maximum transaction",
        "transaction_count_30d": "transactions in past 30 days",
    }
    
    RISK_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.5,
        "high": 0.7,
        "critical": 0.9
    }
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        rag_pipeline: RAGPipeline | None = None,
        config: ExplanationConfig | None = None
    ):
        self.config = config or ExplanationConfig()
        self._llm_provider = llm_provider
        self._rag_pipeline = rag_pipeline
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the explainer with LLM and RAG."""
        if self._initialized:
            return
            
        if self._llm_provider is None:
            self._llm_provider = await get_llm_provider()
            
        if self._rag_pipeline is None and self.config.include_rag_context:
            self._rag_pipeline = RAGPipeline()
            await self._rag_pipeline.initialize()
            
            # Initialize with fraud knowledge
            knowledge_base = FraudKnowledgeBase()
            await knowledge_base.load_to_pipeline(self._rag_pipeline)
            
        self._initialized = True
        logger.info("FraudExplainer initialized")
        
    async def explain(
        self,
        transaction_id: str,
        prediction: float,
        feature_values: dict[str, float],
        feature_importances: dict[str, float] | None = None,
        shap_values: dict[str, float] | None = None,
        model_name: str = "ensemble",
        explanation_type: ExplanationType | None = None
    ) -> FraudExplanation:
        """
        Generate a comprehensive explanation for a fraud prediction.
        
        Args:
            transaction_id: Unique identifier for the transaction
            prediction: Fraud probability (0-1)
            feature_values: Dictionary of feature name -> value
            feature_importances: Optional feature importance scores
            shap_values: Optional SHAP values for the features
            model_name: Name of the model that made the prediction
            explanation_type: Type of explanation to generate
            
        Returns:
            FraudExplanation with full analysis
        """
        await self.initialize()
        
        exp_type = explanation_type or self.config.explanation_type
        
        # Determine risk level
        risk_level = self._calculate_risk_level(prediction)
        is_fraud = prediction >= 0.5
        
        # Analyze feature contributions
        contributions = self._analyze_features(
            feature_values,
            feature_importances or {},
            shap_values or {}
        )
        
        # Get top contributing features
        top_features = contributions[:self.config.top_k_features]
        
        # Generate feature summary
        feature_summary = self._generate_feature_summary(top_features)
        
        # Detect patterns using RAG
        detected_patterns, pattern_confidence = await self._detect_patterns(
            feature_values, prediction, top_features
        )
        
        # Generate natural language explanations
        brief_explanation = await self._generate_brief_explanation(
            prediction, risk_level, top_features, detected_patterns
        )
        
        detailed_explanation = await self._generate_detailed_explanation(
            transaction_id, prediction, risk_level, top_features,
            detected_patterns, exp_type, feature_values
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level, detected_patterns, is_fraud
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(
            prediction, len(top_features), pattern_confidence
        )
        
        return FraudExplanation(
            transaction_id=transaction_id,
            prediction=prediction,
            is_fraud=is_fraud,
            risk_level=risk_level,
            top_features=top_features,
            feature_summary=feature_summary,
            detected_patterns=detected_patterns,
            pattern_confidence=pattern_confidence,
            brief_explanation=brief_explanation,
            detailed_explanation=detailed_explanation,
            recommended_actions=recommendations,
            model_name=model_name,
            explanation_type=exp_type,
            confidence_score=confidence_score
        )
    
    def _calculate_risk_level(self, prediction: float) -> str:
        """Map prediction probability to risk level."""
        if prediction >= self.RISK_THRESHOLDS["critical"]:
            return "critical"
        elif prediction >= self.RISK_THRESHOLDS["high"]:
            return "high"
        elif prediction >= self.RISK_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"
    
    def _analyze_features(
        self,
        feature_values: dict[str, float],
        feature_importances: dict[str, float],
        shap_values: dict[str, float]
    ) -> list[FeatureContribution]:
        """Analyze features and their contributions to the prediction."""
        contributions = []
        
        # Use SHAP values if available, otherwise use feature importances
        scores = shap_values if shap_values else feature_importances
        
        if not scores:
            # Generate approximate importance based on feature values
            scores = self._estimate_importance(feature_values)
        
        for i, (feature, score) in enumerate(
            sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
        ):
            value = feature_values.get(feature, 0.0)
            direction = "increases" if score > 0 else "decreases"
            
            contributions.append(FeatureContribution(
                feature_name=feature,
                value=value,
                contribution=abs(score),
                direction=direction,
                importance_rank=i + 1
            ))
        
        return contributions
    
    def _estimate_importance(self, feature_values: dict[str, float]) -> dict[str, float]:
        """Estimate feature importance when not provided."""
        # Use heuristics for common fraud-related features
        importance_weights = {
            "amt": 0.15,
            "amount_zscore": 0.12,
            "velocity_1h": 0.10,
            "velocity_24h": 0.08,
            "distance_from_home": 0.08,
            "merchant_fraud_rate": 0.10,
            "category_fraud_rate": 0.07,
            "time_since_last_trans": 0.06,
            "is_night": 0.05,
            "is_weekend": 0.04,
            "trans_hour": 0.05,
        }
        
        scores = {}
        for feature, value in feature_values.items():
            weight = importance_weights.get(feature, 0.03)
            # Normalize and apply weight
            normalized = value if abs(value) <= 10 else np.sign(value) * 10
            scores[feature] = weight * normalized
            
        return scores
    
    def _generate_feature_summary(
        self, top_features: list[FeatureContribution]
    ) -> str:
        """Generate a summary of top feature contributions."""
        if not top_features:
            return "No significant features identified."
        
        summaries = []
        for feat in top_features[:3]:
            desc = self.FEATURE_DESCRIPTIONS.get(
                feat.feature_name, feat.feature_name.replace("_", " ")
            )
            summaries.append(
                f"{desc} ({feat.value:.2f}) {feat.direction} risk"
            )
        
        return "; ".join(summaries)
    
    async def _detect_patterns(
        self,
        feature_values: dict[str, float],
        prediction: float,
        top_features: list[FeatureContribution]
    ) -> tuple[list[str], float]:
        """Detect fraud patterns using RAG."""
        detected = []
        confidence = 0.0
        
        if not self.config.include_rag_context or not self._rag_pipeline:
            return self._detect_patterns_heuristic(feature_values, prediction)
        
        # Build query from features
        feature_text = ", ".join([
            f"{f.feature_name}={f.value:.2f}" for f in top_features[:5]
        ])
        query = f"Fraud indicators with {feature_text} prediction={prediction:.2f}"
        
        try:
            results = await self._rag_pipeline.retrieve(
                query, top_k=self.config.rag_top_k
            )
            
            # Extract patterns from results
            for result in results:
                if "pattern" in result.document.metadata.get("type", "").lower():
                    detected.append(result.document.content[:100])
                    confidence = max(confidence, result.score)
                    
        except Exception as e:
            logger.warning(f"RAG pattern detection failed: {e}")
            return self._detect_patterns_heuristic(feature_values, prediction)
        
        if not detected:
            return self._detect_patterns_heuristic(feature_values, prediction)
            
        return detected[:3], confidence
    
    def _detect_patterns_heuristic(
        self,
        feature_values: dict[str, float],
        prediction: float
    ) -> tuple[list[str], float]:
        """Detect patterns using heuristics when RAG is unavailable."""
        patterns = []
        confidence = 0.5
        
        # High amount anomaly
        if feature_values.get("amount_zscore", 0) > 2:
            patterns.append("Unusually high transaction amount")
            confidence = max(confidence, 0.7)
            
        # Velocity anomaly
        velocity_1h = feature_values.get("velocity_1h", 0)
        if velocity_1h > 3:
            patterns.append("High transaction velocity in short period")
            confidence = max(confidence, 0.75)
            
        # Geographic anomaly
        distance = feature_values.get("distance_from_home", 0)
        if distance > 100:
            patterns.append("Transaction far from usual location")
            confidence = max(confidence, 0.6)
            
        # Time anomaly
        if feature_values.get("is_night", 0) > 0.5:
            patterns.append("Unusual transaction time (nighttime)")
            confidence = max(confidence, 0.55)
            
        # Merchant risk
        if feature_values.get("merchant_fraud_rate", 0) > 0.1:
            patterns.append("High-risk merchant")
            confidence = max(confidence, 0.65)
            
        if not patterns:
            patterns.append("No clear fraud pattern detected")
            confidence = 0.3
            
        return patterns, confidence
    
    async def _generate_brief_explanation(
        self,
        prediction: float,
        risk_level: str,
        top_features: list[FeatureContribution],
        patterns: list[str]
    ) -> str:
        """Generate a one-line explanation."""
        if not self._llm_provider:
            return self._generate_brief_fallback(prediction, risk_level, patterns)
        
        prompt = f"""Generate a ONE sentence explanation for a fraud prediction.
        
Risk Level: {risk_level}
Fraud Probability: {prediction:.1%}
Key Factors: {', '.join([f.feature_name for f in top_features[:3]])}
Patterns: {', '.join(patterns[:2])}

Output ONLY the explanation sentence, nothing else."""

        try:
            response = await self._llm_provider.generate(
                prompt,
                temperature=0.3,
                max_tokens=100
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM brief explanation failed: {e}")
            return self._generate_brief_fallback(prediction, risk_level, patterns)
    
    def _generate_brief_fallback(
        self,
        prediction: float,
        risk_level: str,
        patterns: list[str]
    ) -> str:
        """Fallback brief explanation without LLM."""
        pattern_text = patterns[0] if patterns else "multiple risk factors"
        return (
            f"{risk_level.capitalize()} risk ({prediction:.1%} probability) "
            f"due to {pattern_text.lower()}."
        )
    
    async def _generate_detailed_explanation(
        self,
        transaction_id: str,
        prediction: float,
        risk_level: str,
        top_features: list[FeatureContribution],
        patterns: list[str],
        explanation_type: ExplanationType,
        feature_values: dict[str, float]
    ) -> str:
        """Generate a comprehensive explanation."""
        if not self._llm_provider:
            return self._generate_detailed_fallback(
                transaction_id, prediction, risk_level, top_features, patterns
            )
        
        # Get RAG context if available
        rag_context = ""
        if self._rag_pipeline and self.config.include_rag_context:
            try:
                query = f"Explain fraud detection with {patterns[0] if patterns else 'anomaly'}"
                results = await self._rag_pipeline.retrieve(query, top_k=2)
                rag_context = "\n".join([r.document.content[:200] for r in results])
            except Exception:
                pass
        
        # Build prompt based on explanation type
        audience = {
            ExplanationType.BRIEF: "general user",
            ExplanationType.DETAILED: "informed user",
            ExplanationType.TECHNICAL: "data scientist",
            ExplanationType.EXECUTIVE: "business executive",
            ExplanationType.REGULATORY: "compliance officer"
        }.get(explanation_type, "general user")
        
        feature_desc = "\n".join([
            f"- {self.FEATURE_DESCRIPTIONS.get(f.feature_name, f.feature_name)}: "
            f"{f.value:.2f} ({f.direction} risk by {f.contribution:.3f})"
            for f in top_features[:5]
        ])
        
        prompt = f"""Generate a detailed fraud prediction explanation for a {audience}.

Transaction ID: {transaction_id}
Fraud Probability: {prediction:.1%}
Risk Level: {risk_level.upper()}
Decision: {"FLAGGED AS FRAUD" if prediction >= 0.5 else "LIKELY LEGITIMATE"}

Key Contributing Factors:
{feature_desc}

Detected Patterns:
{chr(10).join([f"- {p}" for p in patterns])}

{f"Additional Context:{chr(10)}{rag_context}" if rag_context else ""}

Provide a clear, {len(top_features) * 50 + 100}-word explanation covering:
1. What the model detected
2. Why these factors indicate risk
3. Confidence in the assessment

Write in plain English, suitable for the target audience."""

        try:
            response = await self._llm_provider.generate(
                prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM detailed explanation failed: {e}")
            return self._generate_detailed_fallback(
                transaction_id, prediction, risk_level, top_features, patterns
            )
    
    def _generate_detailed_fallback(
        self,
        transaction_id: str,
        prediction: float,
        risk_level: str,
        top_features: list[FeatureContribution],
        patterns: list[str]
    ) -> str:
        """Fallback detailed explanation without LLM."""
        lines = [
            f"## Fraud Analysis for Transaction {transaction_id}",
            "",
            f"**Risk Assessment:** {risk_level.upper()} ({prediction:.1%} probability)",
            f"**Decision:** {'FLAGGED FOR REVIEW' if prediction >= 0.5 else 'APPROVED'}",
            "",
            "### Key Risk Factors:",
        ]
        
        for feat in top_features[:5]:
            desc = self.FEATURE_DESCRIPTIONS.get(
                feat.feature_name, feat.feature_name.replace("_", " ")
            )
            lines.append(
                f"- **{desc.title()}**: Value of {feat.value:.2f} "
                f"{feat.direction} fraud risk (importance: {feat.contribution:.3f})"
            )
        
        lines.extend(["", "### Detected Patterns:"])
        for pattern in patterns:
            lines.append(f"- {pattern}")
        
        lines.extend([
            "",
            "### Recommendation:",
            self._get_risk_recommendation(risk_level, prediction)
        ])
        
        return "\n".join(lines)
    
    def _get_risk_recommendation(self, risk_level: str, prediction: float) -> str:
        """Get recommendation based on risk level."""
        if risk_level == "critical":
            return "IMMEDIATE ACTION REQUIRED: Block transaction and alert fraud team."
        elif risk_level == "high":
            return "Manual review recommended before processing."
        elif risk_level == "medium":
            return "Consider additional verification (e.g., SMS confirmation)."
        else:
            return "Transaction appears legitimate. Standard processing."
    
    def _generate_recommendations(
        self,
        risk_level: str,
        patterns: list[str],
        is_fraud: bool
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not self.config.include_recommendations:
            return recommendations
        
        if risk_level in ("critical", "high"):
            recommendations.extend([
                "Temporarily hold the transaction",
                "Request additional customer verification",
                "Alert fraud investigation team"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Send SMS/email verification to cardholder",
                "Log for periodic review"
            ])
        else:
            recommendations.append("Process normally with standard monitoring")
        
        # Pattern-specific recommendations
        for pattern in patterns:
            if "velocity" in pattern.lower():
                recommendations.append("Review recent transaction history")
            if "location" in pattern.lower() or "distance" in pattern.lower():
                recommendations.append("Verify customer location")
            if "merchant" in pattern.lower():
                recommendations.append("Flag merchant for review")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_confidence(
        self,
        prediction: float,
        num_features: int,
        pattern_confidence: float
    ) -> float:
        """Calculate overall confidence in the explanation."""
        # Higher when prediction is decisive (close to 0 or 1)
        prediction_confidence = 1 - 4 * (prediction - 0.5) ** 2
        
        # Feature coverage
        feature_confidence = min(num_features / 5, 1.0)
        
        # Weighted average
        confidence = (
            0.4 * prediction_confidence +
            0.3 * feature_confidence +
            0.3 * pattern_confidence
        )
        
        return round(min(max(confidence, 0.0), 1.0), 3)
    
    async def batch_explain(
        self,
        predictions: list[dict[str, Any]],
        batch_size: int = 10
    ) -> list[FraudExplanation]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions: List of dicts with transaction_id, prediction, feature_values, etc.
            batch_size: Number of concurrent explanations
            
        Returns:
            List of FraudExplanations
        """
        await self.initialize()
        
        results = []
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i + batch_size]
            tasks = [
                self.explain(
                    transaction_id=p.get("transaction_id", f"txn_{j}"),
                    prediction=p["prediction"],
                    feature_values=p.get("feature_values", {}),
                    feature_importances=p.get("feature_importances"),
                    shap_values=p.get("shap_values"),
                    model_name=p.get("model_name", "ensemble")
                )
                for j, p in enumerate(batch, start=i)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch explanation failed: {result}")
                else:
                    results.append(result)
        
        return results


async def create_explainer(
    llm_config: LLMConfig | None = None,
    explanation_config: ExplanationConfig | None = None
) -> FraudExplainer:
    """Factory function to create and initialize a FraudExplainer."""
    llm_provider = await get_llm_provider(llm_config)
    explainer = FraudExplainer(
        llm_provider=llm_provider,
        config=explanation_config
    )
    await explainer.initialize()
    return explainer


# Example usage
if __name__ == "__main__":
    async def main():
        explainer = await create_explainer()
        
        # Example transaction
        explanation = await explainer.explain(
            transaction_id="TXN_001",
            prediction=0.87,
            feature_values={
                "amt": 2500.00,
                "amount_zscore": 3.2,
                "velocity_1h": 5,
                "distance_from_home": 150.0,
                "merchant_fraud_rate": 0.12,
                "is_night": 1.0,
                "trans_hour": 2
            },
            model_name="xgboost_fraud_v1"
        )
        
        print(f"Brief: {explanation.brief_explanation}")
        print(f"\nDetailed:\n{explanation.detailed_explanation}")
        print(f"\nRecommendations: {explanation.recommended_actions}")
        
    asyncio.run(main())
