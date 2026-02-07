"""
SentinXFL - Model Registry
===========================

Centralized model registry for creating, loading, and managing models.
Factory pattern for easy model instantiation.

Author: Anshuman Bakshi
"""

from pathlib import Path
from typing import Any, Type

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.ml.base import BaseModel, ModelType

logger = get_logger(__name__)
settings = get_settings()


class ModelRegistry:
    """
    Central registry for all ML models.
    
    Provides:
    - Model type registration
    - Factory creation
    - Model discovery from disk
    - Version management
    """
    
    # Registered model types
    _models: dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model type.
        
        Args:
            model_type: Type identifier (e.g., 'xgboost')
            model_class: Model class to register
        """
        cls._models[model_type] = model_class
        logger.debug(f"Registered model type: {model_type}")
    
    @classmethod
    def get(cls, model_type: str) -> Type[BaseModel]:
        """
        Get model class by type.
        
        Args:
            model_type: Type identifier
            
        Returns:
            Model class
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")
        return cls._models[model_type]
    
    @classmethod
    def create(cls, model_type: str, **kwargs: Any) -> BaseModel:
        """
        Factory method to create model instance.
        
        Args:
            model_type: Type identifier
            **kwargs: Model constructor arguments
            
        Returns:
            Model instance
        """
        model_class = cls.get(model_type)
        return model_class(**kwargs)
    
    @classmethod
    def list_types(cls) -> list[str]:
        """Get list of registered model types."""
        return list(cls._models.keys())
    
    @classmethod
    def load_from_dir(cls, path: Path | str) -> BaseModel:
        """
        Load model from directory, auto-detecting type.
        
        Args:
            path: Path to model directory
            
        Returns:
            Loaded model
        """
        import json
        
        path = Path(path)
        
        # Read metadata to get model type
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {path}")
        
        with open(meta_path) as f:
            metadata = json.load(f)
        
        model_type = metadata.get("model_type")
        if not model_type:
            raise ValueError(f"No model_type in metadata: {path}")
        
        model_class = cls.get(model_type)
        return model_class.load(path)
    
    @classmethod
    def discover_models(cls, directory: Path | str) -> list[dict[str, Any]]:
        """
        Discover all saved models in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of model info dictionaries
        """
        import json
        
        directory = Path(directory)
        models = []
        
        if not directory.exists():
            return models
        
        for subdir in directory.iterdir():
            if not subdir.is_dir():
                continue
            
            meta_path = subdir / "metadata.json"
            if not meta_path.exists():
                continue
            
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                
                models.append({
                    "path": str(subdir),
                    "name": metadata.get("name", subdir.name),
                    "model_type": metadata.get("model_type"),
                    "created_at": metadata.get("created_at"),
                    "trained_at": metadata.get("trained_at"),
                    "metrics": metadata.get("metrics", {}),
                })
            except Exception as e:
                logger.warning(f"Failed to read model metadata from {subdir}: {e}")
        
        return models


# ==========================================
# Register Default Models
# ==========================================

def _register_default_models() -> None:
    """Register all built-in model types."""
    from sentinxfl.ml.xgboost_model import XGBoostModel
    from sentinxfl.ml.lightgbm_model import LightGBMModel
    from sentinxfl.ml.isolation_model import IsolationForestModel
    from sentinxfl.ml.tabnet_model import TabNetModel
    from sentinxfl.ml.ensemble import EnsembleModel
    
    ModelRegistry.register(ModelType.XGBOOST, XGBoostModel)
    ModelRegistry.register(ModelType.LIGHTGBM, LightGBMModel)
    ModelRegistry.register(ModelType.ISOLATION_FOREST, IsolationForestModel)
    ModelRegistry.register(ModelType.TABNET, TabNetModel)
    ModelRegistry.register(ModelType.ENSEMBLE, EnsembleModel)


# Auto-register on import
_register_default_models()


# ==========================================
# Convenience Functions
# ==========================================

def create_model(model_type: str, **kwargs: Any) -> BaseModel:
    """
    Convenience function to create a model.
    
    Args:
        model_type: One of 'xgboost', 'lightgbm', 'isolation_forest', 'tabnet', 'ensemble'
        **kwargs: Model constructor arguments
        
    Returns:
        Model instance
        
    Examples:
        >>> model = create_model('xgboost', name='my_xgb', n_estimators=100)
        >>> model = create_model('tabnet', name='my_tabnet', n_d=32, n_a=32)
    """
    return ModelRegistry.create(model_type, **kwargs)


def load_model(path: Path | str) -> BaseModel:
    """
    Convenience function to load a model from disk.
    
    Auto-detects model type from metadata.
    
    Args:
        path: Path to model directory
        
    Returns:
        Loaded model instance
    """
    return ModelRegistry.load_from_dir(path)


def list_saved_models(directory: Path | str | None = None) -> list[dict[str, Any]]:
    """
    List all saved models in a directory.
    
    Args:
        directory: Directory to search (defaults to settings.models_dir)
        
    Returns:
        List of model info dictionaries
    """
    if directory is None:
        directory = settings.get_absolute_path(settings.models_dir)
    
    return ModelRegistry.discover_models(directory)
