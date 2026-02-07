"""
SentinXFL - Training Pipeline
==============================

Orchestrates model training with stratified cross-validation,
data preprocessing, and comprehensive evaluation.

Author: Anshuman Bakshi
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel
from sentinxfl.ml.metrics import ClassificationMetrics, MetricsCalculator
from sentinxfl.ml.registry import create_model, ModelRegistry

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TrainingResult:
    """Results from a training run."""
    
    model_name: str
    model_type: str
    
    # Metrics
    train_metrics: ClassificationMetrics | None = None
    val_metrics: ClassificationMetrics | None = None
    test_metrics: ClassificationMetrics | None = None
    cv_metrics: list[ClassificationMetrics] = field(default_factory=list)
    
    # Summary stats
    mean_auc_roc: float = 0.0
    std_auc_roc: float = 0.0
    mean_auc_pr: float = 0.0
    std_auc_pr: float = 0.0
    
    # Training info
    training_time_seconds: float = 0.0
    n_samples_train: int = 0
    n_samples_val: int = 0
    n_samples_test: int = 0
    
    # Feature importance
    feature_importance: dict[str, float] = field(default_factory=dict)
    
    # Model path
    model_path: Path | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "train_metrics": self.train_metrics.to_dict() if self.train_metrics else None,
            "val_metrics": self.val_metrics.to_dict() if self.val_metrics else None,
            "test_metrics": self.test_metrics.to_dict() if self.test_metrics else None,
            "mean_auc_roc": self.mean_auc_roc,
            "std_auc_roc": self.std_auc_roc,
            "mean_auc_pr": self.mean_auc_pr,
            "std_auc_pr": self.std_auc_pr,
            "training_time_seconds": self.training_time_seconds,
            "n_samples_train": self.n_samples_train,
            "n_samples_val": self.n_samples_val,
            "n_samples_test": self.n_samples_test,
            "feature_importance_top10": dict(list(self.feature_importance.items())[:10]),
            "model_path": str(self.model_path) if self.model_path else None,
        }


class TrainingPipeline:
    """
    Orchestrates model training with:
    - Stratified train/val/test split
    - Optional cross-validation
    - Model evaluation and comparison
    - Model persistence
    """
    
    def __init__(
        self,
        train_split: float | None = None,
        val_split: float | None = None,
        test_split: float | None = None,
        cv_folds: int = 5,
        random_seed: int | None = None,
    ):
        """
        Initialize training pipeline.
        
        Args:
            train_split: Training set ratio (default from settings)
            val_split: Validation set ratio (default from settings)
            test_split: Test set ratio (default from settings)
            cv_folds: Number of cross-validation folds
            random_seed: Random seed for reproducibility
        """
        self.train_split = train_split or settings.ml_train_split
        self.val_split = val_split or settings.ml_val_split
        self.test_split = test_split or settings.ml_test_split
        self.cv_folds = cv_folds
        self.random_seed = random_seed or settings.ml_random_seed
        
        self._metrics_calc = MetricsCalculator()
        
        # Validate splits
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Splits must sum to 1.0, got {total}")
    
    def prepare_data(
        self,
        df: pl.DataFrame,
        target_col: str,
        feature_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], list[str]]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            feature_cols: List of feature columns (default: all except target)
            exclude_cols: Columns to exclude
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        exclude_cols = exclude_cols or []
        exclude_cols.append(target_col)
        
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Extract features and target
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df.select(target_col).to_numpy().flatten().astype(np.int32)
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        logger.info(f"Prepared data: {X.shape[0]:,} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {np.bincount(y)}")
        
        return X, y, feature_cols
    
    def stratified_split(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
    ) -> tuple[
        tuple[NDArray[np.float32], NDArray[np.int32]],
        tuple[NDArray[np.float32], NDArray[np.int32]],
        tuple[NDArray[np.float32], NDArray[np.int32]],
    ]:
        """
        Perform stratified train/val/test split.
        
        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        test_size = self.test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_seed,
        )
        
        # Second split: train vs val
        val_ratio = self.val_split / (self.train_split + self.val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=self.random_seed,
        )
        
        logger.info(f"Split: train={len(y_train):,}, val={len(y_val):,}, test={len(y_test):,}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_model(
        self,
        model_type: str,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        model_name: str | None = None,
        **model_kwargs: Any,
    ) -> BaseModel:
        """
        Train a single model.
        
        Args:
            model_type: Model type ('xgboost', 'lightgbm', etc.)
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_name: Custom model name
            **model_kwargs: Model hyperparameters
            
        Returns:
            Trained model
        """
        if model_name is None:
            model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model = create_model(model_type, name=model_name, **model_kwargs)
        
        logger.info(f"Training {model_type}: {model_name}")
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        return model
    
    def evaluate_model(
        self,
        model: BaseModel,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        compute_curves: bool = True,
    ) -> ClassificationMetrics:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            compute_curves: Whether to compute ROC/PR curves
            
        Returns:
            Classification metrics
        """
        y_pred = model.predict(X)
        y_proba = model.predict_fraud_probability(X)
        
        return self._metrics_calc.calculate(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            compute_curves=compute_curves,
        )
    
    def cross_validate(
        self,
        model_type: str,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        **model_kwargs: Any,
    ) -> list[ClassificationMetrics]:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            model_type: Model type
            X: Features
            y: Labels
            **model_kwargs: Model hyperparameters
            
        Returns:
            List of metrics for each fold
        """
        kfold = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_seed,
        )
        
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model_name = f"{model_type}_cv_fold{fold}"
            model = self.train_model(
                model_type,
                X_train, y_train,
                X_val, y_val,
                model_name=model_name,
                **model_kwargs,
            )
            
            # Evaluate
            metrics = self.evaluate_model(model, X_val, y_val, compute_curves=False)
            cv_metrics.append(metrics)
            
            logger.info(f"Fold {fold + 1}/{self.cv_folds}: AUC-ROC={metrics.auc_roc:.4f}, AUPRC={metrics.auc_pr:.4f}")
        
        return cv_metrics
    
    def train_and_evaluate(
        self,
        model_type: str,
        df: pl.DataFrame,
        target_col: str,
        feature_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        run_cv: bool = False,
        save_model: bool = True,
        model_name: str | None = None,
        **model_kwargs: Any,
    ) -> TrainingResult:
        """
        Full training pipeline: prepare, split, train, evaluate.
        
        Args:
            model_type: Model type ('xgboost', 'lightgbm', etc.)
            df: Input DataFrame
            target_col: Target column name
            feature_cols: Feature columns (None = auto-detect)
            exclude_cols: Columns to exclude
            run_cv: Run cross-validation
            save_model: Save model to disk
            model_name: Custom model name
            **model_kwargs: Model hyperparameters
            
        Returns:
            TrainingResult with all metrics
        """
        import time
        
        start_time = time.time()
        
        # Prepare data
        X, y, feature_names = self.prepare_data(df, target_col, feature_cols, exclude_cols)
        
        # Split
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.stratified_split(X, y)
        
        result = TrainingResult(
            model_name=model_name or f"{model_type}_model",
            model_type=model_type,
            n_samples_train=len(y_train),
            n_samples_val=len(y_val),
            n_samples_test=len(y_test),
        )
        
        # Cross-validation (optional)
        if run_cv:
            logger.info(f"Running {self.cv_folds}-fold cross-validation")
            cv_metrics = self.cross_validate(model_type, X_train, y_train, **model_kwargs)
            result.cv_metrics = cv_metrics
            
            # Summary stats
            auc_rocs = [m.auc_roc for m in cv_metrics]
            auc_prs = [m.auc_pr for m in cv_metrics]
            result.mean_auc_roc = float(np.mean(auc_rocs))
            result.std_auc_roc = float(np.std(auc_rocs))
            result.mean_auc_pr = float(np.mean(auc_prs))
            result.std_auc_pr = float(np.std(auc_prs))
        
        # Train final model on full train+val
        logger.info("Training final model")
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        model = self.train_model(
            model_type,
            X_train_full, y_train_full,
            model_name=result.model_name,
            **model_kwargs,
        )
        
        # Store feature names
        model.metadata.feature_names = feature_names
        
        # Evaluate on train
        result.train_metrics = self.evaluate_model(model, X_train_full, y_train_full, compute_curves=False)
        
        # Evaluate on test
        result.test_metrics = self.evaluate_model(model, X_test, y_test, compute_curves=True)
        
        # Feature importance
        try:
            result.feature_importance = model.get_feature_importance()
        except (NotImplementedError, RuntimeError):
            pass
        
        # Training time
        result.training_time_seconds = time.time() - start_time
        
        # Save model
        if save_model:
            model_dir = settings.get_absolute_path(settings.models_dir) / result.model_name
            result.model_path = model.save(model_dir)
        
        # Log summary
        logger.info(f"Training complete in {result.training_time_seconds:.1f}s")
        if result.test_metrics:
            logger.info(f"Test metrics: AUC-ROC={result.test_metrics.auc_roc:.4f}, AUPRC={result.test_metrics.auc_pr:.4f}")
            logger.info(result.test_metrics.summary())
        
        return result
    
    def compare_models(
        self,
        df: pl.DataFrame,
        target_col: str,
        model_types: list[str] | None = None,
        feature_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ) -> dict[str, TrainingResult]:
        """
        Train and compare multiple model types.
        
        Args:
            df: Input DataFrame
            target_col: Target column
            model_types: List of model types to compare (default: all)
            feature_cols: Feature columns
            exclude_cols: Columns to exclude
            
        Returns:
            Dictionary mapping model type to TrainingResult
        """
        if model_types is None:
            model_types = ["xgboost", "lightgbm", "isolation_forest"]
        
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n{'='*50}\nTraining {model_type}\n{'='*50}")
            
            try:
                result = self.train_and_evaluate(
                    model_type=model_type,
                    df=df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    exclude_cols=exclude_cols,
                    run_cv=True,
                    save_model=True,
                )
                results[model_type] = result
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                continue
        
        # Print comparison
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON")
        logger.info("="*60)
        
        for model_type, result in sorted(results.items(), 
                                         key=lambda x: x[1].test_metrics.auc_pr if x[1].test_metrics else 0,
                                         reverse=True):
            if result.test_metrics:
                logger.info(
                    f"{model_type:20} | "
                    f"AUC-ROC: {result.test_metrics.auc_roc:.4f} | "
                    f"AUPRC: {result.test_metrics.auc_pr:.4f} | "
                    f"F1: {result.test_metrics.f1_score:.4f}"
                )
        
        return results
