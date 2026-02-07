"""
SentinXFL Data API Routes
==========================

REST API endpoints for data management operations.

Author: Anshuman Bakshi
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from sentinxfl.core.logging import get_logger
from sentinxfl.data.loader import DataLoader
from sentinxfl.data.schemas import DatasetType, DatasetStats, UnifiedDatasetStats
from sentinxfl.core.config import settings

log = get_logger(__name__)
router = APIRouter()


# ============================================
# Request/Response Models
# ============================================
class LoadDatasetRequest(BaseModel):
    """Request to load a dataset."""
    dataset_type: DatasetType
    sample_frac: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction to sample (0.0-1.0)"
    )


class DatasetStatsResponse(BaseModel):
    """Dataset statistics response."""
    name: str
    dataset_type: str
    total_rows: int
    fraud_count: int
    legitimate_count: int
    fraud_ratio: float
    num_features: int
    memory_mb: float


class UnifiedStatsResponse(BaseModel):
    """Unified statistics for all loaded datasets."""
    total_rows: int
    total_fraud: int
    total_legitimate: int
    overall_fraud_ratio: float
    combined_features: int
    total_memory_mb: float
    datasets: list[DatasetStatsResponse]


class DataPreviewResponse(BaseModel):
    """Preview of dataset columns and sample rows."""
    columns: list[str]
    dtypes: dict[str, str]
    sample_rows: list[dict]
    total_rows: int


# Global loader instance
_loader: Optional[DataLoader] = None


def get_loader() -> DataLoader:
    """Get or create data loader instance."""
    global _loader
    if _loader is None:
        _loader = DataLoader()
        _loader.connect()
    return _loader


# ============================================
# Endpoints
# ============================================
@router.get("/datasets", response_model=list[str])
async def list_available_datasets():
    """List available datasets."""
    return [dt.value for dt in DatasetType]


@router.post("/load", response_model=DatasetStatsResponse)
async def load_dataset(request: LoadDatasetRequest):
    """
    Load a dataset into memory.

    - **dataset_type**: Type of dataset to load
    - **sample_frac**: Optional fraction to sample (0.0-1.0)
    """
    loader = get_loader()

    try:
        if request.dataset_type == DatasetType.BANK_ACCOUNT_FRAUD:
            df = loader.load_bank_account_fraud(sample_frac=request.sample_frac)
        elif request.dataset_type == DatasetType.CREDIT_CARD_FRAUD:
            df = loader.load_credit_card_fraud(sample_frac=request.sample_frac)
        elif request.dataset_type == DatasetType.PAYSIM:
            df = loader.load_paysim(sample_frac=request.sample_frac)
        else:
            raise HTTPException(400, f"Unknown dataset type: {request.dataset_type}")

        stats = loader._dataset_stats.get(request.dataset_type)
        if not stats:
            raise HTTPException(500, "Failed to calculate dataset stats")

        return DatasetStatsResponse(
            name=stats.name,
            dataset_type=stats.dataset_type.value,
            total_rows=stats.total_rows,
            fraud_count=stats.fraud_count,
            legitimate_count=stats.legitimate_count,
            fraud_ratio=stats.fraud_ratio,
            num_features=stats.num_features,
            memory_mb=stats.memory_mb,
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.error(f"Error loading dataset: {e}")
        raise HTTPException(500, f"Failed to load dataset: {e}")


@router.post("/load-all", response_model=UnifiedStatsResponse)
async def load_all_datasets(sample_frac: Optional[float] = None):
    """
    Load all available datasets.

    - **sample_frac**: Optional fraction to sample from each dataset
    """
    loader = get_loader()

    try:
        loader.load_all_datasets(sample_frac=sample_frac)
        stats = loader.get_unified_stats()

        return UnifiedStatsResponse(
            total_rows=stats.total_rows,
            total_fraud=stats.total_fraud,
            total_legitimate=stats.total_legitimate,
            overall_fraud_ratio=stats.overall_fraud_ratio,
            combined_features=stats.combined_features,
            total_memory_mb=stats.total_memory_mb,
            datasets=[
                DatasetStatsResponse(
                    name=ds.name,
                    dataset_type=ds.dataset_type.value,
                    total_rows=ds.total_rows,
                    fraud_count=ds.fraud_count,
                    legitimate_count=ds.legitimate_count,
                    fraud_ratio=ds.fraud_ratio,
                    num_features=ds.num_features,
                    memory_mb=ds.memory_mb,
                )
                for ds in stats.datasets
            ],
        )

    except Exception as e:
        log.error(f"Error loading datasets: {e}")
        raise HTTPException(500, f"Failed to load datasets: {e}")


@router.get("/preview/{dataset_type}", response_model=DataPreviewResponse)
async def preview_dataset(dataset_type: DatasetType, n_rows: int = 5):
    """
    Get a preview of a dataset.

    - **dataset_type**: Type of dataset to preview
    - **n_rows**: Number of sample rows (max 100)
    """
    loader = get_loader()
    n_rows = min(n_rows, 100)

    try:
        # Load with small sample for preview
        if dataset_type == DatasetType.BANK_ACCOUNT_FRAUD:
            df = loader.load_bank_account_fraud(sample_frac=0.001)
        elif dataset_type == DatasetType.CREDIT_CARD_FRAUD:
            df = loader.load_credit_card_fraud(sample_frac=0.01)
        elif dataset_type == DatasetType.PAYSIM:
            df = loader.load_paysim(sample_frac=0.001)
        else:
            raise HTTPException(400, f"Unknown dataset type: {dataset_type}")

        return DataPreviewResponse(
            columns=df.columns,
            dtypes={col: str(df[col].dtype) for col in df.columns},
            sample_rows=df.head(n_rows).to_dicts(),
            total_rows=len(df),
        )

    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        log.error(f"Error previewing dataset: {e}")
        raise HTTPException(500, f"Failed to preview dataset: {e}")


@router.get("/stats")
async def get_current_stats():
    """Get statistics for currently loaded datasets."""
    loader = get_loader()

    try:
        if not loader._dataset_stats:
            return {"message": "No datasets loaded yet", "datasets": []}

        stats = loader.get_unified_stats()
        return {
            "total_rows": stats.total_rows,
            "total_fraud": stats.total_fraud,
            "fraud_ratio": stats.overall_fraud_ratio,
            "total_memory_mb": stats.total_memory_mb,
            "datasets_loaded": len(stats.datasets),
        }

    except Exception as e:
        log.error(f"Error getting stats: {e}")
        raise HTTPException(500, str(e))


@router.get("/columns/{dataset_type}")
async def get_dataset_columns(dataset_type: DatasetType):
    """Get column information for a dataset type."""
    from sentinxfl.data.loader import DataLoader

    mappings = DataLoader.COLUMN_MAPPINGS.get(dataset_type, {})
    return {
        "dataset_type": dataset_type.value,
        "column_mappings": mappings,
        "num_columns": len(mappings),
    }
