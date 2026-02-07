"""Federated Learning module for SentinXFL."""

from sentinxfl.fl.aggregators import (
    BaseAggregator,
    FedAvgAggregator,
    MultiKrumAggregator,
    TrimmedMeanAggregator,
    CoordinateMedianAggregator,
    BulyanAggregator,
    ByzantineAttack,
    create_aggregator,
)
from sentinxfl.fl.client import (
    ClientConfig,
    FraudDetectionClient,
    XGBoostFLClient,
    create_client,
    start_client,
)
from sentinxfl.fl.server import (
    ServerConfig,
    DPFedAvg,
    MultiKrumStrategy,
    TrimmedMeanStrategy,
    CoordinateMedianStrategy,
    FLServerManager,
    create_strategy,
    start_server,
)
from sentinxfl.fl.simulator import (
    SimulatedClient,
    SimulationConfig,
    FLSimulator,
    quick_simulate,
)

__all__ = [
    # Aggregators
    "BaseAggregator",
    "FedAvgAggregator",
    "MultiKrumAggregator",
    "TrimmedMeanAggregator",
    "CoordinateMedianAggregator",
    "BulyanAggregator",
    "ByzantineAttack",
    "create_aggregator",
    # Client
    "ClientConfig",
    "FraudDetectionClient",
    "XGBoostFLClient",
    "create_client",
    "start_client",
    # Server
    "ServerConfig",
    "DPFedAvg",
    "MultiKrumStrategy",
    "TrimmedMeanStrategy",
    "CoordinateMedianStrategy",
    "FLServerManager",
    "create_strategy",
    "start_server",
    # Simulator
    "SimulatedClient",
    "SimulationConfig",
    "FLSimulator",
    "quick_simulate",
]
