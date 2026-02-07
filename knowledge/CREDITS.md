# SentinXFL - Credits & Attribution

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Project**: SentinXFL - Privacy-First Federated Fraud Detection

---

## Project Information

| Field | Value |
|-------|-------|
| **Project Title** | SentinXFL: A Privacy-Preserving Federated Learning Framework for Multi-Bank Fraud Detection |
| **Institution** | SRM Institute of Science and Technology, Chennai |
| **Department** | Computer Science & Engineering (Software Engineering) |
| **Project Type** | B.Tech Final Year Major Project |
| **Supervisor** | Dr. Kiruthika |
| **Timeline** | February 2026 - May 2026 |

---

## Team Members

### Anshuman Bakshi — Project Lead & Primary Developer
**Registration Number**: RA2211033010117

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTRIBUTION: 87%                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ██████████████████████████████████████████████████████████     │
│                                                                 │
│  218 / 233 Story Points                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Role**: Full System Design, Architecture & Implementation

**Responsibilities:**
- **System Architecture**: Complete system design and architecture
- **Backend Development**: All FastAPI endpoints and business logic
- **Data Engineering**: DuckDB integration, data pipeline, data loading
- **PII Pipeline**: Complete 5-gate privacy blocking system (PATENT CORE)
- **ML Models**: XGBoost, LightGBM, IsolationForest, TabNet, Ensemble
- **Federated Learning**: Flower server/client, all aggregation strategies
- **Differential Privacy**: All DP mechanisms, RDP accountant, budget management
- **LLM Integration**: Phi-3 setup, RAG pipeline, report generation
- **Hallucination Guards**: NLI verification system
- **Frontend Development**: Complete Next.js dashboard implementation
- **API Design**: All REST API endpoints and schemas
- **Testing**: Unit tests, integration tests, end-to-end tests
- **Documentation**: Technical documentation, API docs, architecture docs
- **Demo & Presentation**: Demo script, presentation slides

**Patent Contributions (100% authorship on all claims):**
1. Certified Data Sanitization Pipeline
2. Byzantine-Robust Federated Aggregation with Fairness Constraints
3. Composable Privacy Budget Across Federated Nodes
4. Grounded Explainable AI with Hallucination Verification
5. Multi-Fraud-Type Unified Detection Framework

**Files Authored:**
- All files in `src/sentinxfl/` directory
- All files in `frontend/` directory
- All API routes and schemas
- All ML model implementations
- All FL and DP implementations
- All LLM and RAG code
- Core documentation files

---

### Komal — Contributor
**Registration Number**: RA2211033010114

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTRIBUTION: 13%                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ████████                                                       │
│                                                                 │
│  15 / 233 Story Points                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Role**: Support & Documentation

**Responsibilities:**
- **UI Research**: Collecting dashboard design references and inspiration
- **Manual Testing**: Browser compatibility testing for dashboard
- **Documentation Formatting**: Formatting and polishing documentation
- **Presentation Support**: Assisting with presentation template research

**Tasks Completed:**
| Sprint | Task | Points |
|--------|------|--------|
| 1 | UI Reference Research | 3 |
| 5 | Dashboard Manual Testing | 3 |
| 6 | Documentation Formatting | 5 |
| 6 | Presentation Research Support | 4 |
| **Total** | | **15** |

**Note**: Komal's contributions are limited to non-technical support tasks. No algorithmic, architectural, or implementation work was performed by Komal.

---

## Contribution Matrix

### By Component

| Component | Anshuman | Komal |
|-----------|----------|-------|
| System Architecture | 100% | 0% |
| Backend (FastAPI) | 100% | 0% |
| Data Pipeline | 100% | 0% |
| PII Blocking Pipeline | 100% | 0% |
| ML Models | 100% | 0% |
| Federated Learning | 100% | 0% |
| Differential Privacy | 100% | 0% |
| LLM & RAG | 100% | 0% |
| Frontend (Next.js) | 100% | 0% |
| UI Research | 0% | 100% |
| Manual Testing | 0% | 100% |
| Technical Documentation | 100% | 0% |
| Documentation Formatting | 20% | 80% |
| Presentation Content | 100% | 0% |
| Presentation Formatting | 50% | 50% |

### By Sprint

| Sprint | Anshuman | Komal |
|--------|----------|-------|
| Sprint 1: Foundation | 42 pts | 3 pts |
| Sprint 2: ML Models | 40 pts | 0 pts |
| Sprint 3: FL + DP | 40 pts | 0 pts |
| Sprint 4: LLM + RAG | 35 pts | 0 pts |
| Sprint 5: Dashboard | 35 pts | 3 pts |
| Sprint 6: Integration | 26 pts | 9 pts |
| **Total** | **218 pts** | **15 pts** |

---

## Intellectual Property Attribution

### Patent Claims — Sole Inventor: Anshuman Bakshi

The following novel contributions are solely attributed to **Anshuman Bakshi**:

#### Claim 1: Certified Data Sanitization Pipeline
```
Inventor: Anshuman Bakshi
Description: A 5-gate privacy pipeline that detects, analyzes, transforms,
             certifies, and audits data to ensure zero PII leakage.
Innovation: Combines statistical detection, quasi-identifier analysis,
            utility-preserving transforms, and cryptographic certification.
```

#### Claim 2: Byzantine-Robust Federated Aggregation
```
Inventor: Anshuman Bakshi
Description: Federated aggregation that tolerates Byzantine (malicious)
             participants while maintaining fairness constraints.
Innovation: Combines Multi-Krum with Trimmed Mean and adds
            participation tracking for fairness.
```

#### Claim 3: Composable Privacy Budget Across Nodes
```
Inventor: Anshuman Bakshi
Description: Privacy budget tracking using RDP composition that works
             across federated nodes with heterogeneous operations.
Innovation: Per-node budgets with aggregate tracking and enforcement.
```

#### Claim 4: Grounded Explainable AI with Verification
```
Inventor: Anshuman Bakshi
Description: LLM-generated explanations grounded in actual model evidence
             with NLI-based hallucination detection.
Innovation: Combines SHAP, TabNet attention, RAG, and NLI verification.
```

#### Claim 5: Multi-Fraud-Type Unified Framework
```
Inventor: Anshuman Bakshi
Description: Single framework detecting account fraud, card fraud,
             and mobile money fraud across datasets.
Innovation: Unified schema with dataset-specific preprocessing
            and ensemble approach.
```

### Publication Attribution

For any publications arising from this project:

**Primary Author**: Anshuman Bakshi  
**Contributing Author**: Komal  
**Corresponding Author**: Anshuman Bakshi  
**Supervisor**: Dr. Kiruthika

**Suggested Author Order**: Anshuman Bakshi, Komal, Dr. Kiruthika

---

## Code Attribution

### External Dependencies

The project uses the following open-source libraries (with appropriate licenses):

| Library | License | Usage |
|---------|---------|-------|
| FastAPI | MIT | Backend framework |
| DuckDB | MIT | Database engine |
| Polars | MIT | DataFrame library |
| XGBoost | Apache 2.0 | ML model |
| LightGBM | MIT | ML model |
| PyTorch | BSD | Deep learning |
| PyTorch-TabNet | MIT | TabNet model |
| Flower | Apache 2.0 | Federated learning |
| Opacus | Apache 2.0 | Differential privacy |
| SHAP | MIT | Explainability |
| Transformers | Apache 2.0 | LLM integration |
| LangChain | MIT | RAG pipeline |
| ChromaDB | Apache 2.0 | Vector store |
| Next.js | MIT | Frontend framework |
| Tailwind CSS | MIT | Styling |
| shadcn/ui | MIT | UI components |
| Recharts | MIT | Charts |

### Dataset Attribution

| Dataset | Source | License |
|---------|--------|---------|
| Bank Account Fraud | Kaggle (NVS Yashwanth) | CC0 |
| Credit Card Fraud | Kaggle (MLG-ULB) | ODbL |
| PaySim | Kaggle (Edgar Lopez-Rojas) | CC BY-SA 4.0 |

---

## Acknowledgments

Special thanks to:

- **Dr. Kiruthika** — Project Supervisor, for guidance and mentorship
- **SRM Institute of Science and Technology** — For providing resources and infrastructure
- **Open Source Community** — For the amazing libraries and tools
- **Kaggle Dataset Authors** — For making fraud detection research possible

---

## Contact

For questions about this project:

**Anshuman Bakshi**  
Email: [anshuman.bakshi@srmist.edu.in]  
Registration: RA2211033010117

---

## Legal Notice

This document serves as an official record of contributions to the SentinXFL project. The attribution percentages and task assignments documented herein reflect the actual work performed by each team member.

All patent claims and novel technical contributions are the intellectual property of **Anshuman Bakshi** as the sole inventor.

---

*Document Version: 2.0 | Date: February 5, 2026 | Prepared by: Anshuman Bakshi*
