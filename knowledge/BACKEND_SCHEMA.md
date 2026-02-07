# SentinXFL - Backend Schema

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)  

---

## 1. API Endpoints

### 1.1 Data Endpoints

```yaml
# Data Management API
/api/v1/data:
  
  GET /datasets:
    description: List available datasets
    response:
      - id: string
        name: string
        rows: integer
        columns: integer
        fraud_rate: float
        status: "loaded" | "processing" | "error"
  
  POST /datasets/load:
    description: Load a dataset from file
    request:
      path: string
      dataset_type: "bank_account" | "credit_card" | "paysim"
    response:
      dataset_id: string
      rows_loaded: integer
      status: string
  
  GET /datasets/{id}/schema:
    description: Get dataset schema
    response:
      columns: Column[]
      pii_columns: string[]
      target_column: string
  
  GET /datasets/{id}/stats:
    description: Get dataset statistics
    response:
      row_count: integer
      column_count: integer
      fraud_count: integer
      fraud_rate: float
      missing_values: dict
  
  POST /datasets/{id}/split:
    description: Create train/val/test split
    request:
      train_ratio: float (default: 0.7)
      val_ratio: float (default: 0.15)
      test_ratio: float (default: 0.15)
      method: "temporal" | "random" | "stratified"
    response:
      train_size: integer
      val_size: integer
      test_size: integer
```

### 1.2 PII Endpoints

```yaml
# PII Blocking API
/api/v1/pii:
  
  POST /detect:
    description: Detect PII in dataset
    request:
      dataset_id: string
    response:
      pii_columns: PIIColumn[]
      quasi_identifiers: string[]
      risk_score: float
  
  POST /certify:
    description: Certify dataset as PII-safe
    request:
      dataset_id: string
      transformations: Transformation[]
    response:
      certificate_id: string
      status: "certified" | "blocked"
      blocked_columns: string[]
      risk_score: float
  
  GET /certificates/{id}:
    description: Get certification details
    response:
      certificate_id: string
      timestamp: datetime
      schema_hash: string
      k_anonymity: integer
      reidentification_risk: float
      transformations_applied: Transformation[]
      status: string
  
  GET /audit:
    description: Get PII audit trail
    query:
      start_date: datetime
      end_date: datetime
      limit: integer
    response:
      entries: AuditEntry[]
```

### 1.3 Model Endpoints

```yaml
# Model Management API
/api/v1/models:
  
  GET /list:
    description: List all models
    response:
      models: Model[]
  
  POST /train:
    description: Train a model
    request:
      model_type: "xgboost" | "lightgbm" | "isolation_forest" | "tabnet" | "ensemble"
      dataset_id: string
      config: ModelConfig
    response:
      job_id: string
      status: "started" | "running" | "completed" | "failed"
  
  GET /train/{job_id}/status:
    description: Get training status
    response:
      job_id: string
      status: string
      progress: float
      current_epoch: integer
      metrics: Metrics
  
  POST /predict:
    description: Make predictions
    request:
      model_id: string
      data: array | dataset_id
    response:
      predictions: float[]
      probabilities: float[]
      latency_ms: float
  
  GET /{model_id}/metrics:
    description: Get model metrics
    response:
      auc_roc: float
      precision: float
      recall: float
      f1_score: float
      confusion_matrix: int[][]
      latency_p50_ms: float
      latency_p99_ms: float
  
  GET /{model_id}/feature_importance:
    description: Get feature importance
    response:
      features: FeatureImportance[]
  
  POST /{model_id}/explain:
    description: Explain a prediction
    request:
      instance: dict
    response:
      prediction: float
      probability: float
      shap_values: dict
      top_features: FeatureContribution[]
```

### 1.4 Federated Learning Endpoints

```yaml
# Federated Learning API
/api/v1/fl:
  
  POST /server/start:
    description: Start FL server
    request:
      config: FLServerConfig
    response:
      server_id: string
      address: string
      status: string
  
  POST /server/stop:
    description: Stop FL server
    request:
      server_id: string
    response:
      status: string
  
  GET /server/status:
    description: Get FL server status
    response:
      server_id: string
      status: "running" | "stopped"
      current_round: integer
      total_rounds: integer
      connected_clients: integer
      aggregation_method: string
  
  POST /client/register:
    description: Register FL client
    request:
      client_id: string
      dataset_id: string
    response:
      client_id: string
      status: string
  
  GET /rounds:
    description: Get FL round history
    response:
      rounds: FLRound[]
  
  POST /simulate:
    description: Run FL simulation (single machine)
    request:
      num_clients: integer
      num_rounds: integer
      dataset_id: string
      aggregation: "fedavg" | "krum" | "trimmed_mean" | "coordinate_median"
    response:
      simulation_id: string
      status: string
```

### 1.5 Differential Privacy Endpoints

```yaml
# Differential Privacy API
/api/v1/dp:
  
  GET /budget:
    description: Get current privacy budget
    response:
      epsilon_total: float
      epsilon_spent: float
      epsilon_remaining: float
      delta: float
      rounds_completed: integer
  
  POST /query:
    description: Execute a DP query
    request:
      query_type: "count" | "sum" | "mean" | "histogram"
      dataset_id: string
      column: string
      epsilon: float
    response:
      result: float | dict
      noise_added: float
      epsilon_used: float
      remaining_budget: float
  
  GET /history:
    description: Get DP query history
    response:
      queries: DPQuery[]
  
  POST /reset:
    description: Reset privacy budget (admin only)
    request:
      new_epsilon: float
      new_delta: float
    response:
      status: string
```

### 1.6 LLM & Report Endpoints

```yaml
# LLM & Reports API
/api/v1/reports:
  
  POST /generate:
    description: Generate a report
    request:
      report_type: "executive" | "evidence" | "technical"
      context:
        metrics: dict
        patterns: Pattern[]
        date_range: DateRange
    response:
      report_id: string
      content: string
      confidence: float
      citations: Citation[]
  
  GET /{report_id}:
    description: Get report by ID
    response:
      report_id: string
      type: string
      content: string
      generated_at: datetime
      confidence: float
  
  GET /list:
    description: List all reports
    query:
      type: string
      start_date: datetime
      end_date: datetime
    response:
      reports: ReportSummary[]
  
  POST /patterns/add:
    description: Add pattern to knowledge base
    request:
      pattern_type: string
      description: string
      indicators: string[]
      severity: "low" | "medium" | "high" | "critical"
    response:
      pattern_id: string
      status: string
  
  GET /patterns:
    description: Get all patterns
    response:
      patterns: Pattern[]
```

### 1.7 Metrics & Health Endpoints

```yaml
# Metrics & Health API
/api/v1/metrics:
  
  GET /dashboard:
    description: Get dashboard metrics
    response:
      transactions_analyzed: integer
      alerts_raised: integer
      fraud_detected: integer
      estimated_exposure: float
      risk_score: float
      risk_level: "low" | "medium" | "high" | "critical"
  
  GET /roi:
    description: Get ROI metrics
    response:
      savings: float
      recovery: float
      false_positive_cost: float
      net_benefit: float
  
  GET /changes:
    description: Get recent changes/alerts
    query:
      limit: integer
    response:
      changes: Change[]
  
  GET /risk-distribution:
    description: Get risk distribution
    response:
      critical: float
      high: float
      medium: float
      low: float
  
  GET /timeline:
    description: Get fraud timeline
    query:
      period: "day" | "week" | "month"
      metric: "count" | "amount" | "rate"
    response:
      data: TimeSeriesPoint[]

/api/v1/health:
  
  GET /status:
    description: Health check
    response:
      status: "healthy" | "degraded" | "unhealthy"
      components:
        database: "up" | "down"
        models: "up" | "down"
        llm: "up" | "down"
        fl_server: "up" | "down"
```

---

## 2. Pydantic Models

### 2.1 Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

# Enums
class DatasetType(str, Enum):
    BANK_ACCOUNT = "bank_account"
    CREDIT_CARD = "credit_card"
    PAYSIM = "paysim"

class SplitMethod(str, Enum):
    TEMPORAL = "temporal"
    RANDOM = "random"
    STRATIFIED = "stratified"

class PIIType(str, Enum):
    DIRECT = "direct"          # SSN, CC#, Name
    QUASI = "quasi_identifier" # Age, ZIP, Gender
    SENSITIVE = "sensitive"    # Health, Financial
    SAFE = "safe"              # Non-PII

class TransformType(str, Enum):
    BIN = "bin"
    GENERALIZE = "generalize"
    SUPPRESS = "suppress"
    DP_NOISE = "dp_noise"
    TOKENIZE = "tokenize"

# Data Models
class Column(BaseModel):
    name: str
    dtype: str
    nullable: bool
    unique_count: int
    null_count: int
    sample_values: List[Any]

class PIIColumn(BaseModel):
    name: str
    pii_type: PIIType
    confidence: float = Field(ge=0, le=1)
    detection_method: str
    recommended_transform: TransformType

class Transformation(BaseModel):
    column: str
    transform_type: TransformType
    params: Optional[Dict[str, Any]] = None

class DatasetStats(BaseModel):
    row_count: int
    column_count: int
    fraud_count: int
    fraud_rate: float = Field(ge=0, le=1)
    missing_values: Dict[str, int]
    memory_mb: float

class DataSplit(BaseModel):
    train_size: int
    val_size: int
    test_size: int
    split_method: SplitMethod
    timestamp_column: Optional[str] = None
```

### 2.2 PII Models

```python
class PIICertificate(BaseModel):
    certificate_id: str
    timestamp: datetime
    dataset_id: str
    schema_hash: str
    k_anonymity: int = Field(ge=1)
    reidentification_risk: float = Field(ge=0, le=1)
    transformations_applied: List[Transformation]
    columns_blocked: List[str]
    status: str  # "CERTIFIED_SAFE" | "BLOCKED"

class AuditEntry(BaseModel):
    entry_id: str
    timestamp: datetime
    action: str  # "DETECT", "TRANSFORM", "CERTIFY", "BLOCK"
    dataset_id: str
    details: Dict[str, Any]
    prev_hash: str
    current_hash: str
```

### 2.3 Model Models

```python
class ModelType(str, Enum):
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ISOLATION_FOREST = "isolation_forest"
    TABNET = "tabnet"
    ENSEMBLE = "ensemble"

class ModelConfig(BaseModel):
    model_type: ModelType
    params: Dict[str, Any] = {}
    
class Metrics(BaseModel):
    auc_roc: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1_score: float = Field(ge=0, le=1)
    accuracy: float = Field(ge=0, le=1)
    latency_p50_ms: float = Field(ge=0)
    latency_p99_ms: float = Field(ge=0)

class FeatureImportance(BaseModel):
    feature: str
    importance: float = Field(ge=0)
    rank: int = Field(ge=1)

class FeatureContribution(BaseModel):
    feature: str
    value: Any
    contribution: float
    direction: str  # "positive" | "negative"

class Model(BaseModel):
    model_id: str
    model_type: ModelType
    created_at: datetime
    metrics: Optional[Metrics] = None
    config: ModelConfig
    status: str  # "training" | "ready" | "failed"

class TrainingJob(BaseModel):
    job_id: str
    model_type: ModelType
    dataset_id: str
    status: str
    progress: float = Field(ge=0, le=1)
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    metrics: Optional[Metrics] = None
```

### 2.4 FL Models

```python
class AggregationType(str, Enum):
    FEDAVG = "fedavg"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    COORDINATE_MEDIAN = "coordinate_median"

class FLServerConfig(BaseModel):
    num_rounds: int = Field(ge=1, default=10)
    min_clients: int = Field(ge=2, default=2)
    aggregation: AggregationType = AggregationType.FEDAVG
    byzantine_robust: bool = False
    byzantine_f: int = Field(ge=0, default=1)

class FLClientStatus(BaseModel):
    client_id: str
    status: str  # "connected" | "training" | "idle" | "disconnected"
    last_update: datetime
    samples_count: int
    current_round: Optional[int] = None

class FLRound(BaseModel):
    round_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    participating_clients: int
    aggregation_method: AggregationType
    epsilon_used: float
    global_metrics: Optional[Metrics] = None

class FLServerStatus(BaseModel):
    server_id: str
    status: str  # "running" | "stopped"
    address: str
    current_round: int
    total_rounds: int
    connected_clients: List[FLClientStatus]
    aggregation_method: AggregationType
```

### 2.5 DP Models

```python
class DPMechanism(str, Enum):
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"

class PrivacyBudget(BaseModel):
    epsilon_total: float = Field(gt=0)
    epsilon_spent: float = Field(ge=0)
    epsilon_remaining: float = Field(ge=0)
    delta: float = Field(gt=0, lt=1)
    mechanism: DPMechanism

class DPQuery(BaseModel):
    query_id: str
    timestamp: datetime
    query_type: str
    column: str
    epsilon_used: float
    noise_added: float
    result: Any

class DPReleaseConfig(BaseModel):
    epsilon: float = Field(gt=0)
    delta: float = Field(gt=0, lt=1)
    clip_norm: float = Field(gt=0, default=1.0)
    mechanism: DPMechanism = DPMechanism.GAUSSIAN
```

### 2.6 Report Models

```python
class ReportType(str, Enum):
    EXECUTIVE = "executive"
    EVIDENCE = "evidence"
    TECHNICAL = "technical"

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Pattern(BaseModel):
    pattern_id: str
    pattern_type: str
    description: str
    indicators: List[str]
    severity: Severity
    frequency: int
    first_seen: datetime
    last_seen: datetime

class Citation(BaseModel):
    claim: str
    evidence: str
    confidence: float = Field(ge=0, le=1)
    source: str

class Report(BaseModel):
    report_id: str
    report_type: ReportType
    content: str
    generated_at: datetime
    confidence: float = Field(ge=0, le=1)
    citations: List[Citation]
    patterns_referenced: List[str]

class ReportRequest(BaseModel):
    report_type: ReportType
    metrics: Optional[Dict[str, Any]] = None
    patterns: Optional[List[str]] = None
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    include_technical: bool = False
```

### 2.7 Dashboard Models

```python
class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class DashboardMetrics(BaseModel):
    transactions_analyzed: int
    alerts_raised: int
    fraud_detected: int
    total_cases: int
    estimated_exposure: float
    loss_exposure: float
    recovered: float
    risk_score: int = Field(ge=0, le=100)
    risk_level: RiskLevel

class ROIMetrics(BaseModel):
    savings_high: float
    savings_high_percent: float
    total_value: float
    total_percent: float

class Change(BaseModel):
    change_id: str
    title: str
    description: str
    severity: Severity
    change_type: str  # "increase" | "decrease" | "new"
    value: str
    timestamp: datetime

class RiskDistribution(BaseModel):
    critical: float = Field(ge=0, le=1)
    high: float = Field(ge=0, le=1)
    medium: float = Field(ge=0, le=1)
    low: float = Field(ge=0, le=1)

class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    value: float
    label: Optional[str] = None
```

---

## 3. Database Schema (DuckDB)

### 3.1 Tables

```sql
-- Datasets metadata
CREATE TABLE datasets (
    dataset_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    dataset_type VARCHAR NOT NULL,  -- 'bank_account', 'credit_card', 'paysim'
    row_count INTEGER,
    column_count INTEGER,
    fraud_count INTEGER,
    fraud_rate DOUBLE,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR DEFAULT 'loaded'
);

-- PII Certificates
CREATE TABLE pii_certificates (
    certificate_id VARCHAR PRIMARY KEY,
    dataset_id VARCHAR REFERENCES datasets(dataset_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_hash VARCHAR NOT NULL,
    k_anonymity INTEGER NOT NULL,
    reidentification_risk DOUBLE NOT NULL,
    transformations JSON,
    columns_blocked JSON,
    status VARCHAR NOT NULL
);

-- Audit Trail (append-only)
CREATE TABLE audit_trail (
    entry_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action VARCHAR NOT NULL,
    dataset_id VARCHAR,
    details JSON,
    prev_hash VARCHAR,
    current_hash VARCHAR NOT NULL
);

-- Models
CREATE TABLE models (
    model_id VARCHAR PRIMARY KEY,
    model_type VARCHAR NOT NULL,
    dataset_id VARCHAR REFERENCES datasets(dataset_id),
    config JSON,
    metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR DEFAULT 'training',
    model_path VARCHAR
);

-- FL Rounds
CREATE TABLE fl_rounds (
    round_id VARCHAR PRIMARY KEY,
    round_number INTEGER NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    participating_clients INTEGER,
    aggregation_method VARCHAR,
    epsilon_used DOUBLE,
    global_metrics JSON
);

-- DP Budget
CREATE TABLE dp_budget (
    id INTEGER PRIMARY KEY DEFAULT 1,
    epsilon_total DOUBLE NOT NULL,
    epsilon_spent DOUBLE DEFAULT 0,
    delta DOUBLE NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- DP Queries
CREATE TABLE dp_queries (
    query_id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_type VARCHAR NOT NULL,
    column_name VARCHAR,
    epsilon_used DOUBLE NOT NULL,
    noise_added DOUBLE,
    result JSON
);

-- Reports
CREATE TABLE reports (
    report_id VARCHAR PRIMARY KEY,
    report_type VARCHAR NOT NULL,
    content TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence DOUBLE,
    citations JSON,
    patterns_referenced JSON
);

-- Patterns (Knowledge Base)
CREATE TABLE patterns (
    pattern_id VARCHAR PRIMARY KEY,
    pattern_type VARCHAR NOT NULL,
    description TEXT,
    indicators JSON,
    severity VARCHAR,
    frequency INTEGER DEFAULT 1,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding BLOB  -- For vector search
);
```

---

## 4. Configuration Schema

### 4.1 Main Configuration (config.yaml)

```yaml
# SentinXFL Configuration
version: "2.0"

# Application settings
app:
  name: "SentinXFL"
  environment: "development"  # development | staging | production
  debug: true
  log_level: "INFO"

# API settings
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"

# Data settings
data:
  base_path: "./data"
  raw_path: "./data/raw"
  processed_path: "./data/processed"
  max_rows_in_memory: 1000000
  chunk_size: 10000

# PII settings
pii:
  enabled: true
  k_anonymity: 5
  max_reidentification_risk: 0.05
  entropy_threshold: 4.0
  cardinality_threshold: 0.9
  patterns:
    credit_card: '^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$'
    ssn: '^\d{3}-\d{2}-\d{4}$'
    email: '^[\w\.-]+@[\w\.-]+\.\w+$'
    phone: '^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'

# Model settings
models:
  default_ensemble_weights:
    xgboost: 0.30
    lightgbm: 0.30
    tabnet: 0.25
    isolation_forest: 0.15
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    device: "cpu"
    n_jobs: 4
  
  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    device: "cpu"
    n_jobs: 4
  
  tabnet:
    n_d: 8
    n_a: 8
    n_steps: 3
    batch_size: 1024
    max_epochs: 50
    device: "cuda"
  
  isolation_forest:
    n_estimators: 100
    contamination: 0.01
    n_jobs: 4

# Federated Learning settings
fl:
  enabled: true
  server:
    address: "0.0.0.0:8080"
    num_rounds: 10
    min_clients: 2
  aggregation:
    default: "fedavg"
    byzantine_robust: false
    byzantine_f: 1

# Differential Privacy settings
dp:
  enabled: true
  epsilon: 1.0
  delta: 1e-5
  mechanism: "gaussian"
  clip_norm: 1.0

# LLM settings
llm:
  enabled: true
  model_id: "microsoft/Phi-3-mini-4k-instruct"
  device: "cuda"
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
  generation:
    max_new_tokens: 512
    temperature: 0.7
    top_p: 0.9

# Vector store settings
vector_store:
  provider: "chromadb"
  persist_directory: "./runs/chroma_db"
  collection_name: "fraud_patterns"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# Runs/output settings
runs:
  base_path: "./runs"
  keep_last_n: 10

# Database settings
database:
  provider: "duckdb"
  path: "./runs/sentinxfl.duckdb"
  memory_limit: "4GB"
  threads: 4
```

---

## 5. Run Artifacts Structure

```
runs/
├── sentinxfl.duckdb           # Main database
├── chroma_db/                 # Vector store
│   └── ...
│
├── run_20260205_103000/       # Individual run
│   ├── run_metadata.json      # Run configuration
│   ├── metrics.json           # Model metrics
│   ├── privacy.json           # DP budget tracking
│   ├── pii_certificate.json   # PII certification
│   │
│   ├── models/                # Trained models
│   │   ├── xgboost.json
│   │   ├── lightgbm.txt
│   │   ├── isolation_forest.pkl
│   │   ├── tabnet.pt
│   │   └── ensemble_weights.json
│   │
│   ├── reports/               # Generated reports
│   │   ├── executive_20260205.md
│   │   └── evidence_20260205.md
│   │
│   └── audit/                 # Audit trail
│       └── audit_trail.jsonl
│
└── run_20260204_150000/       # Previous run
    └── ...
```

---

## 6. Error Codes

```python
class ErrorCode(str, Enum):
    # Data errors (1xxx)
    DATA_NOT_FOUND = "E1001"
    DATA_INVALID_SCHEMA = "E1002"
    DATA_LOAD_FAILED = "E1003"
    DATA_SPLIT_FAILED = "E1004"
    
    # PII errors (2xxx)
    PII_DETECTION_FAILED = "E2001"
    PII_CERTIFICATION_BLOCKED = "E2002"
    PII_HIGH_RISK = "E2003"
    PII_TRANSFORM_FAILED = "E2004"
    
    # Model errors (3xxx)
    MODEL_NOT_FOUND = "E3001"
    MODEL_TRAINING_FAILED = "E3002"
    MODEL_PREDICTION_FAILED = "E3003"
    MODEL_LOAD_FAILED = "E3004"
    
    # FL errors (4xxx)
    FL_SERVER_START_FAILED = "E4001"
    FL_CLIENT_CONNECTION_FAILED = "E4002"
    FL_AGGREGATION_FAILED = "E4003"
    FL_ROUND_TIMEOUT = "E4004"
    
    # DP errors (5xxx)
    DP_BUDGET_EXHAUSTED = "E5001"
    DP_INVALID_QUERY = "E5002"
    DP_NOISE_CALCULATION_FAILED = "E5003"
    
    # LLM errors (6xxx)
    LLM_LOAD_FAILED = "E6001"
    LLM_GENERATION_FAILED = "E6002"
    LLM_HALLUCINATION_DETECTED = "E6003"
    LLM_LOW_CONFIDENCE = "E6004"
    
    # System errors (9xxx)
    INTERNAL_ERROR = "E9001"
    DATABASE_ERROR = "E9002"
    OUT_OF_MEMORY = "E9003"
    GPU_ERROR = "E9004"
```

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
