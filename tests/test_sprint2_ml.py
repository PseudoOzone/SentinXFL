"""
SentinXFL - Sprint 2 Tests
===========================

Tests for ML models, metrics, training pipeline.

Author: Anshuman Bakshi
"""

import numpy as np
import pytest
from numpy.typing import NDArray


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def synthetic_data() -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Generate synthetic imbalanced classification data."""
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 10
    fraud_ratio = 0.02  # 2% fraud (highly imbalanced)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Generate legitimate samples (centered at 0)
    X_legit = np.random.randn(n_legit, n_features).astype(np.float32)
    
    # Generate fraud samples (slightly shifted)
    X_fraud = np.random.randn(n_fraud, n_features).astype(np.float32) + 0.5
    
    # Combine
    X = np.vstack([X_legit, X_fraud])
    y = np.array([0] * n_legit + [1] * n_fraud, dtype=np.int32)
    
    # Shuffle
    idx = np.random.permutation(n_samples)
    return X[idx], y[idx]


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names for testing."""
    return [f"feature_{i}" for i in range(10)]


# ============================================
# Metrics Tests
# ============================================

class TestMetrics:
    """Test metrics calculation."""
    
    def test_metrics_calculator_basic(self, synthetic_data: tuple):
        """Test basic metrics calculation."""
        from sentinxfl.ml.metrics import MetricsCalculator
        
        X, y = synthetic_data
        
        # Fake predictions (random)
        np.random.seed(42)
        y_pred = np.random.randint(0, 2, len(y))
        y_proba = np.random.rand(len(y)).astype(np.float32)
        
        calc = MetricsCalculator()
        metrics = calc.calculate(y, y_pred, y_proba)
        
        assert metrics.n_samples == len(y)
        assert metrics.n_positive > 0  # Should have fraud samples
        assert 0 <= metrics.auc_roc <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
    
    def test_metrics_with_perfect_predictions(self, synthetic_data: tuple):
        """Test metrics with perfect predictions."""
        from sentinxfl.ml.metrics import MetricsCalculator
        
        X, y = synthetic_data
        
        calc = MetricsCalculator()
        metrics = calc.calculate(y, y, y.astype(np.float32))  # Perfect prediction
        
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.auc_roc == 1.0
    
    def test_confusion_matrix(self, synthetic_data: tuple):
        """Test confusion matrix values."""
        from sentinxfl.ml.metrics import MetricsCalculator
        
        X, y = synthetic_data
        
        calc = MetricsCalculator()
        metrics = calc.calculate(y, y, y.astype(np.float32))
        
        total = metrics.true_positives + metrics.false_positives + metrics.true_negatives + metrics.false_negatives
        assert total == len(y)
    
    def test_cost_metrics(self, synthetic_data: tuple):
        """Test cost-based metrics."""
        from sentinxfl.ml.metrics import MetricsCalculator
        
        X, y = synthetic_data
        
        calc = MetricsCalculator()
        
        # All zeros prediction
        y_pred = np.zeros_like(y)
        cost = calc.calculate_cost_metrics(y, y_pred, cost_fp=1.0, cost_fn=10.0)
        
        assert "total_cost" in cost
        assert "fn_cost" in cost
        # Should have FN cost (missing frauds)
        assert cost["fn_cost"] > 0


# ============================================
# Model Tests
# ============================================

class TestXGBoostModel:
    """Test XGBoost model."""
    
    def test_create_model(self):
        """Test model creation."""
        from sentinxfl.ml.xgboost_model import XGBoostModel
        
        model = XGBoostModel(name="test_xgb", n_estimators=10)
        
        assert model.name == "test_xgb"
        assert not model.is_fitted
        assert model.metadata.model_type == "xgboost"
    
    def test_fit_predict(self, synthetic_data: tuple, feature_names: list):
        """Test training and prediction."""
        from sentinxfl.ml.xgboost_model import XGBoostModel
        
        X, y = synthetic_data
        
        # Split
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        model = XGBoostModel(
            name="test_xgb",
            n_estimators=10,
            max_depth=3,
            use_gpu=False,
        )
        model.metadata.feature_names = feature_names
        
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
        
        assert model.is_fitted
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        assert len(y_pred) == len(X_test)
        assert y_proba.shape == (len(X_test), 2)
        assert all(0 <= p <= 1 for p in y_proba.flatten())
    
    def test_feature_importance(self, synthetic_data: tuple, feature_names: list):
        """Test feature importance extraction."""
        from sentinxfl.ml.xgboost_model import XGBoostModel
        
        X, y = synthetic_data
        
        model = XGBoostModel(name="test_xgb", n_estimators=10, use_gpu=False)
        model.metadata.feature_names = feature_names
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == len(feature_names)
        assert all(v >= 0 for v in importance.values())


class TestLightGBMModel:
    """Test LightGBM model."""
    
    def test_create_model(self):
        """Test model creation."""
        from sentinxfl.ml.lightgbm_model import LightGBMModel
        
        model = LightGBMModel(name="test_lgb", n_estimators=10)
        
        assert model.name == "test_lgb"
        assert model.metadata.model_type == "lightgbm"
    
    def test_fit_predict(self, synthetic_data: tuple):
        """Test training and prediction."""
        from sentinxfl.ml.lightgbm_model import LightGBMModel
        
        X, y = synthetic_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        model = LightGBMModel(name="test_lgb", n_estimators=10, max_depth=3)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_fraud_probability(X_test)
        
        assert len(y_pred) == len(X_test)
        assert len(y_proba) == len(X_test)
        assert all(0 <= p <= 1 for p in y_proba)


class TestIsolationForest:
    """Test IsolationForest model."""
    
    def test_create_model(self):
        """Test model creation."""
        from sentinxfl.ml.isolation_model import IsolationForestModel
        
        model = IsolationForestModel(name="test_if", n_estimators=50)
        
        assert model.name == "test_if"
        assert model.metadata.model_type == "isolation_forest"
    
    def test_fit_predict(self, synthetic_data: tuple):
        """Test training and prediction."""
        from sentinxfl.ml.isolation_model import IsolationForestModel
        
        X, y = synthetic_data
        
        model = IsolationForestModel(name="test_if", n_estimators=50, contamination=0.02)
        model.fit(X)  # Unsupervised - y not needed
        
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        anomaly_scores = model.get_anomaly_score(X)
        
        assert len(y_pred) == len(X)
        assert y_proba.shape == (len(X), 2)
        assert len(anomaly_scores) == len(X)
    
    def test_anomaly_scores(self, synthetic_data: tuple):
        """Test anomaly score range."""
        from sentinxfl.ml.isolation_model import IsolationForestModel
        
        X, y = synthetic_data
        
        model = IsolationForestModel(name="test_if", n_estimators=50)
        model.fit(X)
        
        proba = model.predict_fraud_probability(X)
        
        # Probabilities should be in [0, 1]
        assert all(0 <= p <= 1 for p in proba)


# ============================================
# Registry Tests
# ============================================

class TestModelRegistry:
    """Test model registry and factory."""
    
    def test_list_types(self):
        """Test listing available model types."""
        from sentinxfl.ml.registry import ModelRegistry
        
        types = ModelRegistry.list_types()
        
        assert "xgboost" in types
        assert "lightgbm" in types
        assert "isolation_forest" in types
        assert "tabnet" in types
        assert "ensemble" in types
    
    def test_create_model(self):
        """Test factory creation."""
        from sentinxfl.ml.registry import create_model
        
        model = create_model("xgboost", name="factory_test", n_estimators=10)
        
        assert model.name == "factory_test"
        assert model.metadata.model_type == "xgboost"
    
    def test_unknown_type_raises(self):
        """Test that unknown type raises error."""
        from sentinxfl.ml.registry import create_model
        
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("unknown_model")


# ============================================
# Pipeline Tests
# ============================================

class TestTrainingPipeline:
    """Test training pipeline."""
    
    def test_stratified_split(self, synthetic_data: tuple):
        """Test stratified data splitting."""
        from sentinxfl.ml.pipeline import TrainingPipeline
        
        X, y = synthetic_data
        
        pipeline = TrainingPipeline(train_split=0.7, val_split=0.15, test_split=0.15)
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.stratified_split(X, y)
        
        # Check sizes
        total = len(y_train) + len(y_val) + len(y_test)
        assert total == len(y)
        
        # Check stratification (fraud ratio should be similar)
        train_fraud_ratio = y_train.sum() / len(y_train)
        test_fraud_ratio = y_test.sum() / len(y_test)
        
        # Should be within 5% of each other
        assert abs(train_fraud_ratio - test_fraud_ratio) < 0.05
    
    def test_invalid_splits_raises(self):
        """Test that invalid splits raise error."""
        from sentinxfl.ml.pipeline import TrainingPipeline
        
        with pytest.raises(ValueError, match="Splits must sum to 1.0"):
            TrainingPipeline(train_split=0.5, val_split=0.1, test_split=0.1)


# ============================================
# Ensemble Tests
# ============================================

class TestEnsemble:
    """Test ensemble model."""
    
    def test_add_models(self, synthetic_data: tuple):
        """Test adding models to ensemble."""
        from sentinxfl.ml.xgboost_model import XGBoostModel
        from sentinxfl.ml.lightgbm_model import LightGBMModel
        from sentinxfl.ml.ensemble import EnsembleModel
        
        X, y = synthetic_data
        
        # Train base models
        xgb = XGBoostModel(name="xgb", n_estimators=10, use_gpu=False)
        xgb.fit(X, y)
        
        lgb = LightGBMModel(name="lgb", n_estimators=10)
        lgb.fit(X, y)
        
        # Create ensemble
        ensemble = EnsembleModel(name="test_ensemble", strategy="weighted_average")
        ensemble.add_model(xgb, weight=1.0)
        ensemble.add_model(lgb, weight=1.0)
        
        assert len(ensemble._models) == 2
        assert len(ensemble._weights) == 2
    
    def test_ensemble_predict(self, synthetic_data: tuple):
        """Test ensemble prediction."""
        from sentinxfl.ml.xgboost_model import XGBoostModel
        from sentinxfl.ml.lightgbm_model import LightGBMModel
        from sentinxfl.ml.ensemble import EnsembleModel
        
        X, y = synthetic_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # Train base models
        xgb = XGBoostModel(name="xgb", n_estimators=10, use_gpu=False)
        xgb.fit(X_train, y_train)
        
        lgb = LightGBMModel(name="lgb", n_estimators=10)
        lgb.fit(X_train, y_train)
        
        # Create and fit ensemble
        ensemble = EnsembleModel(name="test_ensemble")
        ensemble.add_model(xgb)
        ensemble.add_model(lgb)
        ensemble.fit(X_train, y_train, optimize_weights=True)
        
        # Predict
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        assert len(y_pred) == len(X_test)
        assert y_proba.shape == (len(X_test), 2)


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests with real data."""
    
    @pytest.mark.slow
    def test_full_pipeline_xgboost(self):
        """Test full pipeline with XGBoost on real data."""
        import polars as pl
        from sentinxfl.data.loader import DataLoader
        from sentinxfl.ml.pipeline import TrainingPipeline
        
        # Load small sample
        loader = DataLoader()
        df = loader.load_credit_card_fraud(sample_frac=0.01)
        
        if df is None or len(df) == 0:
            pytest.skip("Credit card dataset not available")
        
        # Train
        pipeline = TrainingPipeline()
        # Determine target column (loader renames 'Class' to 'is_fraud')
        target_col = "is_fraud" if "is_fraud" in df.columns else "Class"
        # Select only numeric feature columns
        exclude = {target_col, "dataset_source"}
        feature_cols = [
            c for c, dt in zip(df.columns, df.dtypes)
            if c not in exclude and str(dt) not in ("String", "Utf8", "Boolean", "Categorical")
        ]
        result = pipeline.train_and_evaluate(
            model_type="xgboost",
            df=df,
            target_col=target_col,
            feature_cols=feature_cols,
            run_cv=False,
            save_model=False,
            n_estimators=10,
            use_gpu=False,
        )
        
        assert result.test_metrics is not None
        assert result.test_metrics.auc_roc > 0.5  # Better than random
        assert result.training_time_seconds > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
