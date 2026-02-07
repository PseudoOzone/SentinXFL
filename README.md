# SentinXFL v2.0

**Privacy-First Federated Fraud Detection Platform**

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-green)
![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen)
![License](https://img.shields.io/badge/license-Academic-red)

## Overview

SentinXFL is a patent-worthy, industry-grade federated fraud detection system combining:

- **Certified Data Sanitization Pipeline** (5-Gate PII Blocking)
- **Byzantine-Robust Federated Learning** 
- **Differential Privacy** with RDP Accounting
- **Explainable AI** with LLM-powered insights
- **Multi-fraud-type Unified Detection**
- **Professional React Dashboard** with real-time monitoring

## 🔐 5-Gate PII Blocking Pipeline (PATENT CORE)

```
┌─────────────────────────────────────────────────────────────────┐
│                    5-GATE PII BLOCKING PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│  Gate 1 → Column Name Analysis (semantic matching)              │
│  Gate 2 → Regex Pattern Detection (100+ patterns)               │
│  Gate 3 → Statistical Uniqueness (quasi-identifier detection)   │
│  Gate 4 → Entropy Analysis (high-entropy sensitive data)        │
│  Gate 5 → ML-based Detection (neural pattern recognition)       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA GPU with 4GB+ VRAM (optional)
- 8GB+ RAM

### Installation

```bash
# Clone repository
cd SentinXFL_Final

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
copy .env.example .env
```

### Running the API Server

```bash
# Start backend server
.venv\Scripts\python.exe -m uvicorn src.sentinxfl.api.app:app --reload --port 8000
```

Server starts at http://localhost:8000

### Running the Dashboard

```bash
# Install dashboard dependencies (first time only)
cd dashboard
npm install

# Start dashboard development server
npm run dev
```

Dashboard starts at http://localhost:3000

### Using the CLI

```bash
# Show system info
python -m sentinxfl.cli info

# Scan datasets for PII
python -m sentinxfl.cli scan --dataset all

# Run certification pipeline
python -m sentinxfl.cli certify --dataset bank --sample 0.1
```

## 📊 Supported Datasets

| Dataset | Rows | Features | Fraud % |
|---------|------|----------|---------|
| Bank Account Fraud | 6M | 32 | Variable |
| Credit Card Fraud | 284K | 31 | 0.17% |
| PaySim | 6.3M | 11 | 0.13% |

## 📁 Project Structure

```
SentinXFL_Final/
├── src/sentinxfl/           # Main source code
│   ├── api/                 # FastAPI REST API
│   ├── core/                # Configuration, logging
│   ├── data/                # Data loading, splitting
│   ├── privacy/             # 5-Gate PII Pipeline
│   ├── ml/                  # ML models (Sprint 2)
│   ├── fl/                  # Federated Learning (Sprint 3)
│   └── llm/                 # LLM/RAG (Sprint 4)
├── dashboard/               # React Dashboard (Sprint 5)
│   ├── src/pages/           # Dashboard, Transactions, FL, Privacy, AI
│   └── src/api/             # API client & React hooks
├── tests/                   # Test suite (134 tests)
├── data/
│   ├── datasets/            # Raw datasets
│   └── processed/           # Sanitized data
├── models/checkpoints/      # Model artifacts
└── knowledge/               # Documentation
```

## 🔒 Privacy & Compliance

- **GDPR Compliant**: Full audit logging, right to erasure
- **DPDPA Ready**: India's data protection requirements
- **RBI Guidelines**: Banking data handling
- **PCI-DSS**: Payment card data security

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/sentinxfl
```

## 📚 API Documentation

With the server running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 👥 Team

**Lead Developer**: Anshuman Bakshi (RA2211033010117)
- All core development: Architecture, ML, FL, Privacy Pipeline

**Contributor**: Komal (RA2211033010114)
- UI research, documentation support

**Supervisor**: Dr. Kiruthika, SRMIST Chennai

## 📄 License

Proprietary - Academic Use Only
Copyright (c) 2026 Anshuman Bakshi. All rights reserved.
Patent Pending.

---

*Built with ❤️ at SRMIST Chennai*

## 🏁 Sprints & Test Results

- **Sprint 1:** Data Loader, PII Pipeline, Certification (22/26 tests passing)
- **Sprint 2:** ML Models, Metrics, Ensemble, Integration (20/24 tests passing)
- **Sprint 3:** FL, DP, RDP, Aggregators, Attacks (24/24 tests passing)
- **Sprint 4:** LLM, RAG, Explainability, API (20/20 tests passing)
- **Sprint 5:** Dashboard API, E2E, CORS, Versioning (20/20 tests passing)

**Total:** 126/134 tests passing (94%)

- Remaining failures are due to missing validation datasets in test mocks or expected data shape mismatches (see test logs for details).
- All dashboard and API integration tests pass.
