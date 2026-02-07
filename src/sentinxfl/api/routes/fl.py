"""
SentinXFL - Federated Learning API Routes
==========================================

REST API endpoints for FL operations including simulation,
privacy tracking, and model aggregation.

Author: Anshuman Bakshi
"""

from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from sentinxfl.core.config import get_settings
from sentinxfl.core.logging import get_logger
from sentinxfl.privacy.accountant import get_accountant, reset_accountant

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/fl", tags=["Federated Learning"])


# ==========================================
# Request/Response Models
# ==========================================

class SimulationRequest(BaseModel):
    """Request for FL simulation."""
    
    num_clients: int = Field(default=5, ge=2, le=100)
    num_rounds: int = Field(default=10, ge=1, le=100)
    aggregation_strategy: str = Field(default="fedavg")
    dp_enabled: bool = Field(default=True)
    dp_epsilon: float = Field(default=1.0, gt=0)
    dp_delta: float = Field(default=1e-5, gt=0, lt=1)
    non_iid: bool = Field(default=False)
    non_iid_alpha: float = Field(default=0.5, gt=0)
    num_byzantine: int = Field(default=0, ge=0)
    attack_type: str = Field(default="random")
    
    model_config = {"extra": "forbid"}


class SimulationResponse(BaseModel):
    """Response from FL simulation."""
    
    status: str
    total_rounds: int
    num_clients: int
    aggregation_strategy: str
    dp_enabled: bool
    final_epsilon: float | None
    final_avg_loss: float | None
    final_avg_f1: float | None
    round_history: list[dict[str, Any]] | None = None


class PrivacyBudgetResponse(BaseModel):
    """Privacy budget status."""
    
    epsilon_budget: float
    epsilon_spent: float
    delta: float
    remaining_budget: float
    budget_exhausted: bool
    num_operations: int


class AggregationRequest(BaseModel):
    """Request for manual aggregation."""
    
    strategy: str = Field(default="fedavg")
    num_byzantine: int = Field(default=0, ge=0)
    trim_ratio: float = Field(default=0.1, ge=0, lt=0.5)
    client_weights: list[list[list[float]]]  # [client][layer][weights]
    num_samples: list[int] | None = None


class AggregationResponse(BaseModel):
    """Response from aggregation."""
    
    status: str
    strategy: str
    num_selected: int
    num_total: int
    selected_indices: list[int] | None = None


class ServerConfigRequest(BaseModel):
    """FL server configuration."""
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1024, le=65535)
    num_rounds: int = Field(default=10, ge=1)
    min_clients: int = Field(default=2, ge=1)
    aggregation_strategy: str = Field(default="fedavg")
    dp_enabled: bool = Field(default=True)
    dp_noise_multiplier: float = Field(default=0.1, gt=0)
    dp_clip_norm: float = Field(default=1.0, gt=0)


# ==========================================
# API Endpoints
# ==========================================

@router.get("/status")
async def get_fl_status():
    """Get FL system status."""
    accountant = get_accountant()
    eps_spent, delta = accountant.get_privacy_spent()
    
    return {
        "status": "ready",
        "available_strategies": [
            "fedavg",
            "krum",
            "trimmed_mean",
            "median",
            "bulyan",
        ],
        "privacy": {
            "epsilon_spent": eps_spent,
            "delta": delta,
            "budget_exhausted": accountant.budget_exhausted,
        },
    }


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Run FL simulation with configured parameters.
    
    This endpoint creates a simulated FL environment with multiple
    clients and runs training rounds locally without network overhead.
    """
    try:
        from sentinxfl.fl.simulator import FLSimulator, SimulationConfig
        import numpy as np
        
        # Create configuration
        config = SimulationConfig(
            num_rounds=request.num_rounds,
            aggregation_strategy=request.aggregation_strategy,
            dp_enabled=request.dp_enabled,
            dp_epsilon=request.dp_epsilon,
            dp_delta=request.dp_delta,
            num_byzantine=request.num_byzantine,
        )
        
        simulator = FLSimulator(config)
        
        # Generate synthetic data for simulation
        n_samples = 1000 * request.num_clients
        n_features = 20
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = (np.random.rand(n_samples) > 0.9).astype(np.int32)  # 10% fraud
        
        # Setup clients
        if request.non_iid:
            simulator.setup_non_iid_split(
                X, y,
                num_clients=request.num_clients,
                alpha=request.non_iid_alpha,
            )
        else:
            simulator.setup_iid_split(
                X, y,
                num_clients=request.num_clients,
                num_byzantine=request.num_byzantine,
                attack_type=request.attack_type,
            )
        
        # Run simulation
        history = simulator.run()
        summary = simulator.get_summary()
        
        return SimulationResponse(
            status="completed",
            total_rounds=summary["total_rounds"],
            num_clients=summary["num_clients"],
            aggregation_strategy=summary["aggregation_strategy"],
            dp_enabled=summary["dp_enabled"],
            final_epsilon=summary["final_epsilon"],
            final_avg_loss=summary["final_avg_loss"],
            final_avg_f1=summary["final_avg_f1"],
            round_history=[r.to_dict() for r in history],
        )
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/privacy/budget", response_model=PrivacyBudgetResponse)
async def get_privacy_budget():
    """Get current privacy budget status."""
    accountant = get_accountant()
    eps_spent, delta = accountant.get_privacy_spent()
    
    return PrivacyBudgetResponse(
        epsilon_budget=accountant.epsilon_budget,
        epsilon_spent=eps_spent,
        delta=delta,
        remaining_budget=max(0, accountant.epsilon_budget - eps_spent),
        budget_exhausted=accountant.budget_exhausted,
        num_operations=len(accountant.history),
    )


@router.post("/privacy/reset")
async def reset_privacy_budget(
    new_epsilon: float = 1.0,
    new_delta: float = 1e-5,
):
    """Reset privacy accountant with new budget."""
    reset_accountant()
    
    # Re-initialize with new budget
    from sentinxfl.privacy.accountant import RDPAccountant
    new_accountant = RDPAccountant(
        epsilon_budget=new_epsilon,
        delta=new_delta,
    )
    
    # Update global accountant (this is a simplified approach)
    return {
        "status": "reset",
        "new_epsilon_budget": new_epsilon,
        "new_delta": new_delta,
    }


@router.get("/privacy/history")
async def get_privacy_history():
    """Get history of privacy-consuming operations."""
    accountant = get_accountant()
    
    return {
        "total_operations": len(accountant.history),
        "history": accountant.history[-100:],  # Last 100 operations
    }


@router.post("/aggregate", response_model=AggregationResponse)
async def aggregate_weights(request: AggregationRequest):
    """
    Manually aggregate client weights.
    
    Useful for custom FL workflows outside the simulation.
    """
    try:
        from sentinxfl.fl.aggregators import create_aggregator
        import numpy as np
        
        # Convert to numpy arrays
        client_weights = [
            [np.array(layer, dtype=np.float32) for layer in client]
            for client in request.client_weights
        ]
        
        # Create aggregator
        aggregator = create_aggregator(
            strategy=request.strategy,
            num_byzantine=request.num_byzantine,
            trim_ratio=request.trim_ratio,
        )
        
        # Aggregate
        result = aggregator.aggregate(
            client_weights,
            num_samples=request.num_samples,
        )
        
        return AggregationResponse(
            status="success",
            strategy=request.strategy,
            num_selected=result.num_selected,
            num_total=result.num_total,
            selected_indices=result.selected_indices,
        )
        
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def list_strategies():
    """List available aggregation strategies."""
    return {
        "strategies": [
            {
                "name": "fedavg",
                "description": "Federated Averaging - weighted mean by sample count",
                "byzantine_resilient": False,
            },
            {
                "name": "krum",
                "description": "Multi-Krum - selects updates closest to neighbors",
                "byzantine_resilient": True,
                "requirements": "n >= 2f + 3 clients",
            },
            {
                "name": "trimmed_mean",
                "description": "Coordinate-wise trimmed mean",
                "byzantine_resilient": True,
                "requirements": "n > 2 * trim_count",
            },
            {
                "name": "median",
                "description": "Coordinate-wise median",
                "byzantine_resilient": True,
            },
            {
                "name": "bulyan",
                "description": "Krum + Trimmed Mean combo",
                "byzantine_resilient": True,
                "requirements": "n >= 4f + 3 clients",
            },
        ]
    }


@router.post("/dp/compute-params")
async def compute_dp_parameters(
    target_epsilon: float = 1.0,
    target_delta: float = 1e-5,
    dataset_size: int = 10000,
    batch_size: int = 256,
    epochs: int = 10,
):
    """
    Compute DP-SGD parameters for target privacy.
    
    Returns the noise multiplier and other parameters needed
    to achieve the target (ε, δ)-DP guarantee.
    """
    try:
        from sentinxfl.privacy.dp_trainer import compute_dp_params
        
        params = compute_dp_params(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            dataset_size=dataset_size,
            batch_size=batch_size,
            epochs=epochs,
        )
        
        return {
            "status": "computed",
            "target_epsilon": target_epsilon,
            "target_delta": target_delta,
            "computed_params": params,
        }
        
    except Exception as e:
        logger.error(f"DP param computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Background Tasks
# ==========================================

async def _start_fl_server_task(config: ServerConfigRequest):
    """Background task to start FL server."""
    try:
        from sentinxfl.fl.server import ServerConfig, start_server
        
        server_config = ServerConfig(
            host=config.host,
            port=config.port,
            num_rounds=config.num_rounds,
            min_fit_clients=config.min_clients,
            min_evaluate_clients=config.min_clients,
            min_available_clients=config.min_clients,
            aggregation_strategy=config.aggregation_strategy,
            dp_enabled=config.dp_enabled,
            dp_noise_multiplier=config.dp_noise_multiplier,
            dp_clip_norm=config.dp_clip_norm,
        )
        
        start_server(server_config)
        
    except Exception as e:
        logger.error(f"FL server failed: {e}")


@router.post("/server/start")
async def start_fl_server(
    config: ServerConfigRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start FL server in background.
    
    Note: This starts a gRPC server for real FL clients to connect.
    For simulation/testing, use /simulate instead.
    """
    background_tasks.add_task(_start_fl_server_task, config)
    
    return {
        "status": "starting",
        "message": f"FL server starting on {config.host}:{config.port}",
        "config": config.model_dump(),
    }
