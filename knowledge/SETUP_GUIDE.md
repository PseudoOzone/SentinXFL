# SentinXFL - Environment Setup Guide

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)

---

## 1. Prerequisites

### 1.1 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 6GB | RTX 3050+ 4GB |
| RAM | 8GB | 16GB+ |
| Storage | 20GB free | 50GB+ SSD |
| CPU | 4 cores | 6+ cores |

### 1.2 Software Requirements

| Software | Version | Notes |
|----------|---------|-------|
| OS | Windows 10/11 or Ubuntu 20.04+ | Windows preferred for dev |
| Python | 3.11.x | NOT 3.12+ (LightGBM issues) |
| Node.js | 18.x LTS | For frontend |
| CUDA | 12.1+ | For GPU acceleration |
| Git | 2.40+ | Version control |
| VS Code | Latest | Recommended IDE |

---

## 2. Quick Start

### 2.1 Clone Repository

```powershell
# Clone the repository
git clone https://github.com/yourusername/sentinxfl.git
cd sentinxfl

# Or if starting fresh
cd c:\Users\anshu\SentinXFL_Final
```

### 2.2 One-Command Setup (Windows)

```powershell
# Run setup script
.\scripts\setup.ps1
```

### 2.3 One-Command Setup (Linux/Mac)

```bash
# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

## 3. Detailed Setup

### 3.1 Python Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\.venv\Scripts\activate.bat

# Activate (Linux/Mac)
source .venv/bin/activate

# Verify Python version
python --version  # Should be 3.11.x
```

### 3.2 Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
```

### 3.3 Verify GPU

```powershell
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA: True, Device: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## 4. Configuration

### 4.1 Environment Variables

Create `.env` file in project root:

```bash
# .env file

# Application
APP_ENV=development
APP_DEBUG=true
APP_LOG_LEVEL=DEBUG

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Database
DATABASE_PATH=./data/sentinxfl.duckdb

# Datasets (in project folder)
DATASETS_PATH=./data/datasets/

# ML
DEVICE=cuda
BATCH_SIZE=1024
MAX_VRAM_GB=4

# LLM
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
LLM_QUANTIZATION=4bit
LLM_MAX_TOKENS=2048

# Privacy
DP_EPSILON=1.0
DP_DELTA=1e-5

# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=sentinxfl

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
```

### 4.2 Configuration File

Create `config/config.yaml`:

```yaml
# config/config.yaml

app:
  name: SentinXFL
  version: 2.0.0
  environment: development

data:
  base_path: ${DATASETS_PATH}  # ./data/datasets/
  datasets:
    bank_account:
      files:
        - Base.csv
        - Variant I.csv
        - Variant II.csv
        - Variant III.csv
        - Variant IV.csv
        - Variant V.csv
    credit_card:
      files:
        - creditcard.csv
    paysim:
      files:
        - PS_20174392719_1491204439457_log.csv
  
  processing:
    chunk_size: 100000
    lazy_loading: true

models:
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    
  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    
  tabnet:
    n_d: 8
    n_a: 8
    n_steps: 3
    batch_size: 1024
    max_epochs: 50
    
  ensemble:
    weights:
      xgboost: 0.3
      lightgbm: 0.3
      tabnet: 0.2
      isolation_forest: 0.2

privacy:
  pii:
    entropy_threshold: 4.0
    cardinality_threshold: 0.9
    k_anonymity_min: 5
    
  dp:
    epsilon: 1.0
    delta: 1e-5
    max_gradient_norm: 1.0

fl:
  num_rounds: 10
  min_clients: 2
  fraction_fit: 1.0
  aggregation: "fedavg"  # or "krum", "trimmed_mean", "median"

llm:
  model: "microsoft/Phi-3-mini-4k-instruct"
  quantization: "4bit"
  max_new_tokens: 1024
  temperature: 0.7
  top_p: 0.9
```

---

## 5. Frontend Setup

### 5.1 Install Node.js Dependencies

```powershell
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Or with pnpm (faster)
pnpm install
```

### 5.2 Frontend Environment

Create `frontend/.env.local`:

```bash
# Frontend environment
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=SentinXFL
```

---

## 6. Running the Application

### 6.1 Start Backend

```powershell
# Terminal 1: Start FastAPI server
cd c:\Users\anshu\SentinXFL_Final
.\.venv\Scripts\Activate.ps1
uvicorn src.sentinxfl.main:app --reload --host 0.0.0.0 --port 8000
```

### 6.2 Start Frontend

```powershell
# Terminal 2: Start Next.js dev server
cd c:\Users\anshu\SentinXFL_Final\frontend
npm run dev
```

### 6.3 Access Application

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 7. Development Tools

### 7.1 VS Code Extensions

Install these recommended extensions:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "charliermarsh.ruff",
    "tamasfe.even-better-toml",
    "bradlc.vscode-tailwindcss",
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "GitHub.copilot"
  ]
}
```

### 7.2 VS Code Settings

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.formatting.provider": "none",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "ruff.lint.args": ["--config=pyproject.toml"],
  "editor.rulers": [88, 120],
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.ruff_cache": true
  }
}
```

### 7.3 Pre-commit Hooks

```powershell
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

---

## 8. Database Setup

### 8.1 DuckDB (Auto-created)

DuckDB database is created automatically on first run at `./data/sentinxfl.duckdb`.

### 8.2 ChromaDB (Auto-created)

ChromaDB vector store is created at `./data/chroma/`.

### 8.3 Initialize Data

```powershell
# Run data initialization script
python scripts/init_data.py

# Or via API
curl -X POST http://localhost:8000/api/v1/data/datasets/load \
  -H "Content-Type: application/json" \
  -d '{"path": "c:/Users/anshu/Downloads/raw/", "dataset_type": "bank_account"}'
```

---

## 9. Running Tests

### 9.1 Unit Tests

```powershell
# Run all tests
pytest

# Run with coverage
pytest --cov=src/sentinxfl --cov-report=html

# Run specific module
pytest tests/test_pii.py -v

# Run with markers
pytest -m "not slow"
```

### 9.2 Integration Tests

```powershell
# Run integration tests
pytest tests/integration/ -v
```

### 9.3 E2E Tests

```powershell
# Run end-to-end tests
pytest tests/e2e/ -v --slow
```

---

## 10. Troubleshooting

### 10.1 CUDA Out of Memory

```python
# Add to code before GPU operations
import torch
torch.cuda.empty_cache()

# Or set environment variable
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 10.2 LightGBM Installation Issues

```powershell
# Windows: Install Visual C++ Build Tools first
# Then:
pip install lightgbm --no-binary lightgbm
```

### 10.3 PyTorch Not Using GPU

```powershell
# Reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 10.4 Port Already in Use

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process
taskkill /PID <PID> /F
```

### 10.5 Node.js Module Issues

```powershell
# Clear cache and reinstall
cd frontend
rm -rf node_modules
rm package-lock.json
npm install
```

---

## 11. Project Structure

```
SentinXFL_Final/
├── .env                    # Environment variables (gitignored)
├── .gitignore
├── pyproject.toml          # Python dependencies
├── README.md
├── config/
│   └── config.yaml         # Application config
├── data/                   # Data storage (gitignored)
│   ├── sentinxfl.duckdb
│   └── chroma/
├── frontend/               # Next.js frontend
│   ├── .env.local
│   ├── package.json
│   ├── app/
│   └── components/
├── knowledge/              # Documentation
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   └── ...
├── mlruns/                 # MLflow tracking (gitignored)
├── scripts/
│   ├── setup.ps1
│   ├── setup.sh
│   └── init_data.py
├── src/
│   └── sentinxfl/
│       ├── __init__.py
│       ├── main.py
│       ├── api/
│       ├── data/
│       ├── dp/
│       ├── fl/
│       ├── llm/
│       ├── models/
│       ├── pii/
│       └── rag/
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/
```

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
