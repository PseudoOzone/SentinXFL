# SentinXFL - Technology Stack

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)  
> **Optimized For**: RTX 3050 4GB VRAM, 16GB RAM

---

## 1. Hardware Constraints & Optimization

### 1.1 Target Hardware
| Component | Specification | Constraint |
|-----------|--------------|------------|
| GPU | NVIDIA RTX 3050 | **4GB VRAM** |
| RAM | 16GB DDR4 | **~12GB usable** (OS overhead) |
| CPU | Intel/AMD (6+ cores assumed) | Parallel processing |
| Storage | SSD | Fast I/O for datasets |

### 1.2 Memory Budget

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VRAM BUDGET (4GB Total)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  TabNet Training               ~1.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Phi-3-mini (4-bit)            ~2.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  CUDA Context + Buffers        ~0.5 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Safety Margin                 ~0.5 GB                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  TOTAL:                           ~4.0 GB ✅                        │
│                                                                     │
│  NOTE: TabNet and LLM should NOT run simultaneously!               │
│        - Training phase: TabNet only                                │
│        - Report phase: LLM only                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAM BUDGET (16GB Total)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Operating System              ~4.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  DuckDB (1M rows loaded)       ~2.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  XGBoost/LightGBM Training     ~2.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Python Runtime + Libraries    ~2.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  FastAPI + Workers             ~1.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Next.js Dev Server            ~1.0 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  ChromaDB                      ~0.5 GB                      │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Safety Margin                 ~3.5 GB                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  TOTAL:                          ~16.0 GB ✅                        │
│                                                                     │
│  OPTIMIZATION STRATEGIES:                                           │
│  • Use DuckDB lazy evaluation (don't load all data)                 │
│  • Process datasets sequentially, not in parallel                   │
│  • Clear GPU memory between TabNet and LLM                          │
│  • Use batch processing (chunk_size=10000)                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Backend Stack

### 2.1 Core Python (3.11)

```toml
# pyproject.toml dependencies

[project]
name = "sentinxfl"
version = "2.0.0"
requires-python = ">=3.11,<3.13"  # Not 3.13 (LightGBM compatibility)

[project.dependencies]
# Core
python = "^3.11"
pydantic = "^2.5"
pydantic-settings = "^2.1"

# API
fastapi = "^0.109"
uvicorn = {extras = ["standard"], version = "^0.27"}
python-multipart = "^0.0.6"

# Data Processing
duckdb = "^0.9"
polars = "^0.20"
numpy = "^1.26"
pandas = "^2.1"  # Compatibility only

# Machine Learning (CPU)
scikit-learn = "^1.4"
xgboost = "^2.0"
lightgbm = "^4.2"
shap = "^0.44"

# Deep Learning (GPU - careful with VRAM)
torch = "^2.1"  # CPU fallback available
pytorch-tabnet = "^4.1"

# Federated Learning
flwr = "^1.6"  # Flower framework

# Differential Privacy
opacus = "^1.4"
autodp = "^0.2"

# LLM (4-bit quantized)
transformers = "^4.37"
accelerate = "^0.26"
bitsandbytes = "^0.42"  # 4-bit quantization
peft = "^0.8"  # LoRA
sentencepiece = "^0.1.99"

# Vector Store
chromadb = "^0.4"
sentence-transformers = "^2.3"

# Utilities
orjson = "^3.9"
python-dotenv = "^1.0"
httpx = "^0.26"
loguru = "^0.7"
typer = "^0.9"
rich = "^13.7"

# Experiment Tracking
mlflow = "^2.10"
```

### 2.2 Why Each Choice

| Package | Why Chosen | Alternative Considered |
|---------|------------|----------------------|
| **FastAPI** | Async, fast, auto-docs | Flask (slower), Django (overkill) |
| **DuckDB** | SQL + fast analytics, low memory | SQLite (slower), Pandas (RAM hog) |
| **Polars** | Fast DataFrames, lazy eval | Pandas (slower, more RAM) |
| **XGBoost** | Industry standard, CPU efficient | CatBoost (similar) |
| **LightGBM** | Faster than XGBoost, good accuracy | XGBoost only |
| **TabNet** | Attention mechanism, interpretable | Standard NN (less interpretable) |
| **Flower** | Production FL framework | Raw gRPC (more work) |
| **Opacus** | DP-SGD with PyTorch | TF Privacy (TensorFlow dependency) |
| **Phi-3-mini** | Best quality at 4GB, Microsoft | Llama-2-7B (too big), TinyLlama (weak) |
| **ChromaDB** | Simple, embedded, good enough | Pinecone (cloud), Weaviate (complex) |

---

## 3. Frontend Stack

### 3.1 Next.js 14 Stack

```json
// package.json dependencies
{
  "dependencies": {
    "next": "^14.1",
    "react": "^18.2",
    "react-dom": "^18.2",
    
    // UI Components
    "@radix-ui/react-accordion": "^1.1",
    "@radix-ui/react-alert-dialog": "^1.0",
    "@radix-ui/react-dialog": "^1.0",
    "@radix-ui/react-dropdown-menu": "^2.0",
    "@radix-ui/react-progress": "^1.0",
    "@radix-ui/react-select": "^2.0",
    "@radix-ui/react-tabs": "^1.0",
    "@radix-ui/react-tooltip": "^1.0",
    
    // Styling
    "tailwindcss": "^3.4",
    "class-variance-authority": "^0.7",
    "clsx": "^2.1",
    "tailwind-merge": "^2.2",
    
    // Charts
    "recharts": "^2.10",
    
    // Icons
    "lucide-react": "^0.316",
    
    // Data Fetching
    "@tanstack/react-query": "^5.17",
    "axios": "^1.6",
    
    // Forms
    "react-hook-form": "^7.49",
    "zod": "^3.22",
    "@hookform/resolvers": "^3.3"
  },
  "devDependencies": {
    "typescript": "^5.3",
    "@types/node": "^20.11",
    "@types/react": "^18.2",
    "autoprefixer": "^10.4",
    "postcss": "^8.4",
    "eslint": "^8.56",
    "eslint-config-next": "^14.1"
  }
}
```

### 3.2 Why Each Choice

| Package | Why Chosen | Notes |
|---------|------------|-------|
| **Next.js 14** | App Router, Server Components, Vercel-native | Best for production |
| **Tailwind CSS** | Utility-first, fast development | Industry standard |
| **shadcn/ui** | Beautiful, accessible, customizable | Not a package, copy components |
| **Recharts** | React-native, good docs | Lighter than D3 |
| **Lucide** | Clean icons, tree-shakeable | Better than Font Awesome |
| **TanStack Query** | Best data fetching library | Caching, mutations |

---

## 4. ML Model Specifications

### 4.1 XGBoost Configuration

```python
XGBOOST_CONFIG = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",  # Memory efficient
    "device": "cpu",  # Save GPU for TabNet/LLM
    "n_jobs": 4,  # Don't use all cores
    "random_state": 42
}
```

### 4.2 LightGBM Configuration

```python
LIGHTGBM_CONFIG = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "device": "cpu",
    "n_jobs": 4,
    "random_state": 42,
    "verbose": -1
}
```

### 4.3 IsolationForest Configuration

```python
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 100,
    "contamination": 0.01,  # Expected fraud rate
    "max_samples": "auto",
    "random_state": 42,
    "n_jobs": 4
}
```

### 4.4 TabNet Configuration (GPU)

```python
TABNET_CONFIG = {
    "n_d": 8,  # Reduced from 64 for VRAM
    "n_a": 8,
    "n_steps": 3,
    "gamma": 1.3,
    "n_independent": 1,
    "n_shared": 1,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": {"lr": 2e-2},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 10, "gamma": 0.9},
    "mask_type": "entmax",
    "device_name": "cuda",  # Uses GPU
    "verbose": 1
}

# Batch size optimized for 4GB VRAM
TABNET_FIT_PARAMS = {
    "batch_size": 1024,  # Reduced for VRAM
    "virtual_batch_size": 128,
    "max_epochs": 50,
    "patience": 10
}
```

### 4.5 Ensemble Configuration

```python
ENSEMBLE_CONFIG = {
    "weights": {
        "xgboost": 0.30,
        "lightgbm": 0.30,
        "isolation_forest": 0.15,
        "tabnet": 0.25
    },
    "method": "weighted_average",  # or "stacking"
    "calibration": True  # Platt scaling
}
```

---

## 5. LLM Configuration

### 5.1 Phi-3-mini Setup

```python
LLM_CONFIG = {
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "quantization": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True
    },
    "generation": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1
    },
    "device_map": "cuda:0",
    "torch_dtype": "float16"
}

# Estimated VRAM: ~2GB
```

### 5.2 Embedding Model

```python
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu",  # Keep on CPU to save GPU for LLM
    "normalize_embeddings": True
}

# Model size: ~80MB
# Embedding dimension: 384
```

### 5.3 ChromaDB Configuration

```python
CHROMADB_CONFIG = {
    "persist_directory": "./runs/chroma_db",
    "collection_name": "fraud_patterns",
    "distance_fn": "cosine"
}
```

---

## 6. Federated Learning Configuration

### 6.1 Flower Server

```python
FL_SERVER_CONFIG = {
    "server_address": "0.0.0.0:8080",
    "num_rounds": 10,  # Reduced for faster iteration
    "min_fit_clients": 2,
    "min_evaluate_clients": 2,
    "min_available_clients": 2,
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0
}
```

### 6.2 Flower Client

```python
FL_CLIENT_CONFIG = {
    "server_address": "127.0.0.1:8080",
    "local_epochs": 1,
    "batch_size": 1024
}
```

### 6.3 Aggregation Strategies

```python
AGGREGATION_CONFIG = {
    "default": "fedavg",
    "strategies": {
        "fedavg": {
            "weighted": True
        },
        "krum": {
            "num_to_select": 2,  # Select top 2
            "num_malicious": 1   # Assume 1 Byzantine
        },
        "trimmed_mean": {
            "trim_ratio": 0.1  # Remove 10% outliers
        },
        "coordinate_median": {
            # No parameters
        }
    }
}
```

---

## 7. Differential Privacy Configuration

### 7.1 Privacy Budget

```python
DP_CONFIG = {
    "epsilon": 1.0,  # Total budget
    "delta": 1e-5,
    "max_grad_norm": 1.0,  # Gradient clipping
    "noise_multiplier": 1.1,  # Noise scale
    "accountant": "rdp"  # Rényi DP
}
```

### 7.2 Per-Round Budget

```python
# With 10 FL rounds and ε=1.0:
# ε per round ≈ 0.1 (with RDP composition)

ROUND_BUDGET = {
    "epsilon_per_round": 0.1,
    "auto_scale": True  # Adjust noise based on remaining budget
}
```

---

## 8. Data Processing Configuration

### 8.1 DuckDB Settings

```python
DUCKDB_CONFIG = {
    "memory_limit": "4GB",  # Limit RAM usage
    "threads": 4,
    "temp_directory": "./runs/duckdb_temp"
}
```

### 8.2 Batch Processing

```python
BATCH_CONFIG = {
    "chunk_size": 10000,  # Rows per batch
    "max_workers": 4,  # Parallel workers
    "prefetch_batches": 2
}
```

### 8.3 Data Splits

```python
SPLIT_CONFIG = {
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "method": "temporal",  # Time-based split
    "stratify": True,  # Stratify by fraud label
    "random_state": 42
}
```

---

## 9. Development Tools

### 9.1 Code Quality

```toml
# pyproject.toml

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --cov=src --cov-report=term-missing"
```

### 9.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.14
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic]
```

---

## 10. Resource Usage Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RESOURCE USAGE BY PHASE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PHASE 1: Data Loading + PII Blocking                               │
│  ├── VRAM: 0 GB (CPU only)                                          │
│  ├── RAM:  ~4 GB (DuckDB + Polars)                                  │
│  └── CPU:  4 cores                                                  │
│                                                                     │
│  PHASE 2: Model Training                                            │
│  ├── XGBoost/LightGBM/IsolationForest (sequential)                  │
│  │   ├── VRAM: 0 GB                                                 │
│  │   ├── RAM:  ~6 GB                                                │
│  │   └── CPU:  4 cores                                              │
│  ├── TabNet (after CPU models complete)                             │
│  │   ├── VRAM: ~1 GB                                                │
│  │   ├── RAM:  ~4 GB                                                │
│  │   └── GPU:  RTX 3050                                             │
│                                                                     │
│  PHASE 3: Federated Learning                                        │
│  ├── VRAM: ~1 GB (if TabNet in FL)                                  │
│  ├── RAM:  ~6 GB                                                    │
│  └── Network: Local (simulation)                                    │
│                                                                     │
│  PHASE 4: LLM Report Generation                                     │
│  ├── VRAM: ~2 GB (Phi-3 4-bit)                                      │
│  ├── RAM:  ~4 GB                                                    │
│  └── Note: Run AFTER TabNet unloads                                 │
│                                                                     │
│  PHASE 5: Dashboard                                                 │
│  ├── VRAM: 0 GB                                                     │
│  ├── RAM:  ~2 GB (Next.js + API)                                    │
│  └── CPU:  2 cores                                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. Installation Commands

```bash
# Backend setup
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"

# Frontend setup
cd frontend
npm install

# Run development
# Terminal 1: Backend
cd backend && uvicorn src.sentinxfl.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Flower Server (optional, for FL)
cd backend && python -m sentinxfl.fl.server
```

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
