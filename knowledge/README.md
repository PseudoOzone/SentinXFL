# SentinXFL Knowledge Base

> **Project**: SentinXFL - Privacy-First Federated Fraud Detection  
> **Author**: Anshuman Bakshi (RA2211033010117)  
> **Institution**: SRMIST Chennai  
> **Supervisor**: Dr. Kiruthika

---

## 📚 Documentation Index

| Document | Description | Last Updated |
|----------|-------------|--------------|
| [PRD.md](PRD.md) | Product Requirements Document - functional/non-functional requirements | Feb 5, 2026 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture with detailed ASCII diagrams | Feb 5, 2026 |
| [TECH_STACK.md](TECH_STACK.md) | Technology choices optimized for 4GB VRAM | Feb 5, 2026 |
| [DATASETS.md](DATASETS.md) | Dataset documentation, column mappings, feature engineering | Feb 5, 2026 |
| [BACKEND_SCHEMA.md](BACKEND_SCHEMA.md) | API endpoints, Pydantic models, database schema | Feb 5, 2026 |
| [FRONTEND_GUIDELINES.md](FRONTEND_GUIDELINES.md) | Dark theme design system, component specifications | Feb 5, 2026 |
| [COMPLIANCE.md](COMPLIANCE.md) | GDPR, DPDPA, RBI, PCI-DSS compliance mapping | Feb 5, 2026 |
| [SECURITY.md](SECURITY.md) | Threat model, authentication, encryption, security controls | Feb 5, 2026 |
| [TESTING_STRATEGY.md](TESTING_STRATEGY.md) | Unit, integration, E2E testing approach | Feb 5, 2026 |
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | Environment setup and installation instructions | Feb 5, 2026 |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Sprint-by-sprint development plan | Feb 5, 2026 |
| [JIRA_TASKS.md](JIRA_TASKS.md) | 56 JIRA-style tasks with story points | Feb 5, 2026 |
| [CREDITS.md](CREDITS.md) | Team attribution and contribution breakdown | Feb 5, 2026 |
| [GLOSSARY.md](GLOSSARY.md) | Technical terms, acronyms, and formulas | Feb 5, 2026 |

---

## 🎯 Project Overview

**SentinXFL** is a privacy-first federated fraud detection platform that enables multiple financial institutions to collaboratively train fraud detection models without sharing raw customer data.

### Core Innovation (Patent Claims)

1. **Certified Data Sanitization Pipeline** - 5-gate PII blocking with cryptographic certification
2. **Byzantine-Robust Federated Aggregation** - Tolerates malicious participants
3. **Composable Privacy Budget** - RDP accounting across federated nodes
4. **Grounded Explainable AI** - LLM reports with hallucination verification
5. **Multi-Fraud-Type Detection** - Unified framework for account, card, and payment fraud

### Key Metrics

| Metric | Target |
|--------|--------|
| Total Dataset Size | 12.6M+ rows |
| Model AUC-ROC | > 0.92 |
| Inference Latency | < 100ms |
| Privacy Budget | ε = 1.0, δ = 10⁻⁵ |
| k-Anonymity | k ≥ 5 |

---

## 📅 Timeline

```
Feb 5 ──────────────────────────────────────────────────── May 1
  │                                                          │
  ├── Week 1: Foundation + PII Pipeline                      │
  ├── Week 2: ML Models (XGBoost, LightGBM, TabNet)          │
  ├── Week 3: Federated Learning + Differential Privacy      │
  ├── Week 4: LLM + RAG Report Generation                    │
  ├── Week 5: Dashboard (Next.js)                            │
  ├── Week 6: Integration + Testing                          │
  │                                                          │
  └── Buffer: Publication prep, panel presentation ──────────┘
```

---

## 👥 Team

| Member | Role | Contribution |
|--------|------|--------------|
| **Anshuman Bakshi** | Lead Developer | 87% (All technical work) |
| Komal | Contributor | 13% (UI research, docs, testing) |

---

## 🛠 Quick Start

```bash
# Clone and setup
cd c:\Users\anshu\SentinXFL_Final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Start backend
uvicorn src.sentinxfl.main:app --reload

# Start frontend (new terminal)
cd frontend
npm install
npm run dev
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

---

## 📊 Datasets

| Dataset | Location | Rows | Fraud % |
|---------|----------|------|---------|}
| Bank Account Fraud | `data/datasets/` (6 files) | 6M | 1.4% |
| Credit Card | `data/datasets/creditcard.csv` | 285K | 0.17% |
| PaySim | `data/datasets/PS_*.csv` | 6.3M | 0.13% |

See [DATASETS.md](DATASETS.md) for column mappings and preprocessing.

---

## 🔒 Privacy & Compliance

- **Differential Privacy**: ε=1.0, δ=10⁻⁵ (RDP accounting)
- **k-Anonymity**: Minimum k=5 enforced
- **PII Blocking**: 5-gate certified sanitization
- **Audit Trail**: Hash-chained, tamper-proof logging
- **Compliance**: GDPR, DPDPA 2023, RBI Guidelines, PCI-DSS

See [COMPLIANCE.md](COMPLIANCE.md) and [SECURITY.md](SECURITY.md) for details.

---

## 📝 Documentation Conventions

- All documents use Markdown format
- ASCII diagrams for architecture visualization
- Code examples in Python (backend) and TypeScript (frontend)
- Version tracking at document header
- Author attribution on each document

---

*Knowledge Base Version: 2.0 | Last Updated: February 5, 2026*
