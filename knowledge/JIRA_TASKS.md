# SentinXFL - Task Breakdown (JIRA Style)

> **Version**: 2.0  
> **Last Updated**: February 20, 2026  
> **Project Lead**: Anshuman Bakshi (RA2211033010117)  
> **Team Member**: Komal (RA2211033010114)

---

## Sprint Overview

| Sprint | Dates | Story Points | Focus |
|--------|-------|--------------|-------|
| Sprint 1 | Feb 5-12 | 42 pts | Foundation + PII Pipeline |
| Sprint 2 | Feb 12-19 | 40 pts | ML Models |
| Sprint 3 | Feb 19-26 | 40 pts | FL + Differential Privacy |
| Sprint 4 | Feb 26 - Mar 5 | 35 pts | LLM + RAG |
| Sprint 5 | Mar 5-12 | 48 pts | Dashboard UI |
| Sprint 6 | Mar 12-19 | 34 pts | Integration & Polish |
| **Total** | **6 weeks** | **239 pts** | - |

---

## Team Allocation

```
┌──────────────────────────────────────────────────────────────────┐
│                       TEAM ALLOCATION                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Anshuman Bakshi — Project Lead & Core Engineer                  │
│  ├── System Architecture & Project Setup                         │
│  ├── 5-Gate PII Blocking Pipeline (Patent Core)                  │
│  ├── All ML Models (XGBoost, LightGBM, TabNet, Ensemble)         │
│  ├── Federated Learning (Flower Server/Client, Aggregators)      │
│  ├── Differential Privacy (RDP, Gaussian, Budget Manager)        │
│  ├── LLM & RAG (Phi-3, ChromaDB, Hallucination Guards)          │
│  ├── All Backend API Development (FastAPI)                       │
│  ├── Frontend API Client & React Hooks                           │
│  ├── End-to-End Integration & Demo Pipeline                      │
│  └── Performance Optimization & Deployment                       │
│                                              189 pts (80%)       │
│                                                                  │
│  Komal — UI/Dashboard Developer                                  │
│  ├── React + Vite + Tailwind Project Setup                       │
│  ├── Dashboard Layout, Sidebar & Navigation                      │
│  ├── Executive Overview Dashboard Page                           │
│  ├── Transactions & Local Bank View Page                         │
│  ├── Federated Learning Status Page                              │
│  ├── Privacy & Compliance Page                                   │
│  ├── Explainability & AI Reports Page                            │
│  ├── Login & Authentication UI                                   │
│  ├── Settings & Notifications UI                                 │
│  ├── Cross-Browser Testing & Responsive QA                       │
│  └── Documentation Formatting & User Guide                       │
│                                               50 pts (20%)       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Epic Breakdown

### EPIC-1: Data Infrastructure
**Owner**: Anshuman | **Points**: 16 | **Sprint**: 1

### EPIC-2: PII Blocking Pipeline (PATENT CORE)
**Owner**: Anshuman | **Points**: 26 | **Sprint**: 1

### EPIC-3: ML Model Framework
**Owner**: Anshuman | **Points**: 40 | **Sprint**: 2

### EPIC-4: Differential Privacy
**Owner**: Anshuman | **Points**: 14 | **Sprint**: 3

### EPIC-5: Federated Learning
**Owner**: Anshuman | **Points**: 26 | **Sprint**: 3

### EPIC-6: LLM & RAG Intelligence
**Owner**: Anshuman | **Points**: 35 | **Sprint**: 4

### EPIC-7: Dashboard UI
**Owner**: Komal (UI) + Anshuman (API Client) | **Points**: 48 | **Sprint**: 5

### EPIC-8: Integration & Polish
**Owner**: Anshuman (integration) + Komal (docs) | **Points**: 34 | **Sprint**: 6

---

## Sprint 1: Foundation + PII (Feb 5-12)

### SXFL-001: Project Setup & Repository Init
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Initialize project structure with directory layout, pyproject.toml, git repo, pre-commit hooks, and .env configuration.

**Acceptance Criteria:**
- [x] Project structure created
- [x] pyproject.toml with all dependencies
- [x] .gitignore configured
- [x] Pre-commit hooks set up

---

### SXFL-002: FastAPI Application Skeleton
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create FastAPI application with config management, structured logging, health endpoint, and CORS middleware.

**Acceptance Criteria:**
- [x] FastAPI app runs on localhost:8000
- [x] Health check endpoint returns 200
- [x] Configuration via environment variables
- [x] Structured logging configured

---

### SXFL-003: DuckDB Data Loader
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement lazy CSV loader using DuckDB for Bank Account Fraud (6M rows), Credit Card (285K rows), and PaySim (6.3M rows) datasets with schema detection.

**Acceptance Criteria:**
- [x] Bank Account Fraud dataset loads (6M rows)
- [x] Credit Card dataset loads (285K rows)
- [x] PaySim dataset loads (6.3M rows)
- [x] Schema detection working
- [x] 10+ unit tests passing

---

### SXFL-004: Temporal & Stratified Data Splitter
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement temporal and stratified splitters for train/val/test splits with no data leakage validation.

**Acceptance Criteria:**
- [x] Temporal split working
- [x] Stratified split working
- [x] No data leakage validation
- [x] Configurable ratios

---

### SXFL-005: Data API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 2 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create REST API endpoints for data operations.

**Acceptance Criteria:**
- [x] GET /api/v1/data/datasets
- [x] GET /api/v1/data/{dataset}/schema
- [x] POST /api/v1/data/{dataset}/split
- [x] Integration tests passing

---

### SXFL-006: Statistical PII Detection (Gate 3 & 4)
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement entropy-based and cardinality-based PII detection with confidence scoring. Core of the 5-Gate Certified Data Sanitization Pipeline.

**Acceptance Criteria:**
- [x] Entropy calculation working
- [x] Cardinality ratio calculation
- [x] PII columns auto-detected
- [x] Confidence scores assigned
- [x] 15+ unit tests

---

### SXFL-007: Regex Pattern PII Detection (Gate 2)
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement 100+ regex patterns for known PII formats.

**Acceptance Criteria:**
- [x] Credit card pattern
- [x] SSN pattern
- [x] Email pattern
- [x] Phone pattern
- [x] Indian ID patterns (Aadhaar, PAN)

---

### SXFL-008: k-Anonymity Analyzer (Gate 3)
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement k-anonymity verification and re-identification risk scoring.

**Acceptance Criteria:**
- [x] k-anonymity calculation
- [x] Quasi-identifier combination analysis
- [x] Risk score (1/k)
- [x] Configurable k threshold

---

### SXFL-009: PII Transformations (Gate 5)
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement utility-preserving PII transformations.

**Acceptance Criteria:**
- [x] Binning transform
- [x] Generalization transform
- [x] Suppression transform
- [x] DP noise transform
- [x] Hard blocking gate

---

### SXFL-010: PII Certificate Generator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create certificate generation with schema hashing for certified data sanitization.

**Acceptance Criteria:**
- [x] Certificate generation
- [x] Schema hash verification
- [x] Certificate storage
- [x] Verification endpoint

---

### SXFL-011: Hash-Chained Audit Trail
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement hash-chained audit log with tamper detection for full regulatory compliance.

**Acceptance Criteria:**
- [x] Hash-chained entries
- [x] Tamper detection
- [x] Audit API endpoints
- [x] Query by time range

---

**Sprint 1 Total: 42 pts** (Anshuman: 42 pts)

---

## Sprint 2: ML Models (Feb 12-19)

### SXFL-012: Base Model Interface
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create abstract base class for all models with common interface.

**Acceptance Criteria:**
- [x] fit() interface
- [x] predict() interface
- [x] predict_proba() interface
- [x] save/load interface
- [x] Metrics calculator

---

### SXFL-013: XGBoost Fraud Detector
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement XGBoost wrapper with hyperparameter configuration.

**Acceptance Criteria:**
- [x] XGBoost training works
- [x] Early stopping configured
- [x] Hyperparameter config
- [x] AUC-ROC > 0.90 on all datasets
- [x] Serialization working

---

### SXFL-014: LightGBM Fraud Detector
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement LightGBM wrapper optimized for CPU.

**Acceptance Criteria:**
- [x] LightGBM training works
- [x] CPU-optimized configuration
- [x] Comparable to XGBoost
- [x] Unit tests passing

---

### SXFL-015: IsolationForest Anomaly Detector
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement IsolationForest for anomaly-based detection.

**Acceptance Criteria:**
- [x] IsolationForest training works
- [x] Anomaly scores → probabilities
- [x] Contamination tuned
- [x] Unit tests passing

---

### SXFL-016: TabNet Deep Learning Model
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 8 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement TabNet optimized for 4GB VRAM.

**Acceptance Criteria:**
- [x] TabNet trains without OOM
- [x] VRAM usage ≤ 1GB
- [x] Attention weights extractable
- [x] Performance comparable
- [x] GPU memory management

---

### SXFL-017: Weighted Ensemble Model
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement weighted ensemble combining all models.

**Acceptance Criteria:**
- [x] Weighted averaging
- [x] Weight optimization
- [x] Probability calibration
- [x] Stacking option

---

### SXFL-018: SHAP Explainability Integration
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Integrate SHAP for model explanations.

**Acceptance Criteria:**
- [x] SHAP values computed
- [x] Feature importance ranked
- [x] Explanation API
- [x] Caching for speed

---

### SXFL-019: MLflow Experiment Tracking
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Set up MLflow for experiment tracking.

**Acceptance Criteria:**
- [x] Metrics logged
- [x] Model artifacts stored
- [x] Experiment management

---

### SXFL-020: Model API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create REST API endpoints for model operations.

**Acceptance Criteria:**
- [x] POST /api/v1/models/train
- [x] POST /api/v1/models/predict
- [x] GET /api/v1/models/{id}/metrics
- [x] GET /api/v1/models/{id}/explain

---

**Sprint 2 Total: 40 pts** (Anshuman: 40 pts)

---

## Sprint 3: FL + DP (Feb 19-26)

### SXFL-021: Gaussian Mechanism for DP
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement Gaussian mechanism with correctly calibrated noise and (ε,δ) guarantees.

**Acceptance Criteria:**
- [x] Noise correctly calibrated
- [x] (ε,δ) guarantees
- [x] Configurable parameters
- [x] Unit tests

---

### SXFL-022: Rényi DP Accountant
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement Rényi Differential Privacy accountant.

**Acceptance Criteria:**
- [x] RDP composition
- [x] (ε,δ) converter
- [x] Tight bounds
- [x] Unit tests

---

### SXFL-023: Privacy Budget Manager
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Track and enforce privacy budget across operations.

**Acceptance Criteria:**
- [x] Budget tracking per client
- [x] Aggregate budget tracking
- [x] Budget enforcement
- [x] Query API

---

### SXFL-024: DP API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create REST API for DP operations.

**Acceptance Criteria:**
- [x] GET /api/v1/dp/budget
- [x] POST /api/v1/dp/query
- [x] GET /api/v1/dp/history

---

### SXFL-025: Flower FL Server
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement Flower server with FedAvg aggregation strategy.

**Acceptance Criteria:**
- [x] Server starts
- [x] FedAvg aggregation
- [x] Round management
- [x] Client tracking

---

### SXFL-026: Flower FL Client
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement Flower client with local training and DP integration.

**Acceptance Criteria:**
- [x] Client connects to server
- [x] Local training works
- [x] Gradient extraction
- [x] DP integration

---

### SXFL-027: Multi-Krum Byzantine Aggregator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement Byzantine-robust Multi-Krum aggregation.

**Acceptance Criteria:**
- [x] Multi-Krum implemented
- [x] Tolerates f malicious
- [x] Configurable f
- [x] Unit tests

---

### SXFL-028: Trimmed Mean & Coordinate Median
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement Trimmed Mean and Coordinate Median robust aggregation.

**Acceptance Criteria:**
- [x] Trimmed Mean implemented
- [x] Coordinate Median implemented
- [x] Configurable trimming ratio
- [x] Unit tests

---

### SXFL-029: FL Simulator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Single-machine FL simulation for testing.

**Acceptance Criteria:**
- [x] Simulate 3+ clients
- [x] Data partitioning
- [x] FL converges
- [x] Metrics tracked

---

### SXFL-030: FL API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create REST API for FL operations.

**Acceptance Criteria:**
- [x] POST /api/v1/fl/simulate
- [x] GET /api/v1/fl/status
- [x] GET /api/v1/fl/history

---

**Sprint 3 Total: 40 pts** (Anshuman: 40 pts)

---

## Sprint 4: LLM + RAG (Feb 26 - Mar 5)

### SXFL-031: Phi-3 Model Loading & Quantization
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Load Phi-3-mini-4k with 4-bit quantization, VRAM ≤ 2GB.

**Acceptance Criteria:**
- [x] Model loads
- [x] VRAM ≤ 2GB
- [x] Basic generation works
- [x] Memory management

---

### SXFL-032: Prompt Engineering Templates
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create and tune prompt templates for report generation.

**Acceptance Criteria:**
- [x] Executive summary template
- [x] Evidence report template
- [x] Technical report template
- [x] Template rendering

---

### SXFL-033: ChromaDB Vector Store Setup
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Set up ChromaDB vector store with persistence.

**Acceptance Criteria:**
- [x] ChromaDB running
- [x] Collections created
- [x] Persistence working
- [x] Unit tests

---

### SXFL-034: Embedding Generation Pipeline
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create MiniLM embedding pipeline with batch processing.

**Acceptance Criteria:**
- [x] MiniLM embeddings
- [x] Batch processing
- [x] ChromaDB integration
- [x] Unit tests

---

### SXFL-035: RAG Retriever with MMR Diversity
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement semantic retrieval with Maximal Marginal Relevance diversity.

**Acceptance Criteria:**
- [x] Semantic retrieval
- [x] MMR diversity
- [x] Relevance filtering
- [x] Configurable k

---

### SXFL-036: Evidence Collector for XAI
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Collect SHAP and TabNet evidence for LLM report injection.

**Acceptance Criteria:**
- [x] SHAP values collected
- [x] TabNet attention collected
- [x] Evidence formatted
- [x] Injected into prompts

---

### SXFL-037: NLI Hallucination Guards
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Implement NLI-based hallucination detection.

**Acceptance Criteria:**
- [x] Claim extraction
- [x] Entailment checking
- [x] Confidence scoring
- [x] Warnings generated

---

### SXFL-038: LLM Report Generator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Generate full fraud reports with LLM including evidence integration.

**Acceptance Criteria:**
- [x] Executive summary generated
- [x] Evidence integrated
- [x] PDF export
- [x] Quality validation

---

### SXFL-039: Report API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create REST API for report operations.

**Acceptance Criteria:**
- [x] POST /api/v1/reports/generate
- [x] GET /api/v1/reports/{id}
- [x] GET /api/v1/reports/history

---

**Sprint 4 Total: 35 pts** (Anshuman: 35 pts)

---

## Sprint 5: Dashboard UI (Mar 5-12)

### SXFL-040: React + Vite + Tailwind Project Setup
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Initialize React project with Vite, configure Tailwind CSS, set up dark theme with custom color palette, and install UI component libraries.

**Acceptance Criteria:**
- [x] React + Vite running
- [x] Tailwind CSS configured
- [x] Dark theme applied
- [x] Component library installed

---

### SXFL-041: Dashboard Layout & Navigation
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build main application layout with collapsible sidebar, top header, breadcrumbs, and responsive footer.

**Acceptance Criteria:**
- [x] Main layout complete
- [x] Sidebar navigation with icons
- [x] Header with logo & user info
- [x] Responsive design

---

### SXFL-042: Executive Overview Dashboard Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 8 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build Executive Overview page with KPI stat cards, risk score gauge, risk distribution donut chart, fraud timeline chart, and recent alerts list.

**Acceptance Criteria:**
- [x] KPI cards (Total Transactions, Fraud Rate, Model Accuracy, Privacy Budget)
- [x] Risk score gauge
- [x] Risk distribution donut chart
- [x] Fraud timeline chart
- [x] Recent alerts list
- [x] API integration

---

### SXFL-043: Transactions & Local Bank View Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build Transactions page with model comparison table, performance charts, and feature importance.

**Acceptance Criteria:**
- [x] Model comparison table
- [x] Performance bar chart
- [x] Confusion matrix visualization
- [x] Feature importance chart
- [x] Training history

---

### SXFL-044: Federated Learning Status Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build FL Training page showing federation status, round progress, and convergence.

**Acceptance Criteria:**
- [x] Federation status overview
- [x] Round progress tracker
- [x] Client contributions display
- [x] Aggregation metrics
- [x] Convergence charts

---

### SXFL-045: Privacy & Compliance Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build Privacy page with privacy budget tracker, PII audit display, and compliance status.

**Acceptance Criteria:**
- [x] Privacy budget tracker visualization
- [x] PII audit display
- [x] Compliance status indicators
- [x] DP parameter controls

---

### SXFL-046: Explainability & AI Reports Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build Explainability page with LLM-generated report display, SHAP charts, and export.

**Acceptance Criteria:**
- [x] LLM-generated fraud reports display
- [x] SHAP waterfall charts
- [x] Feature importance visualizations
- [x] Report export functionality

---

### SXFL-047: Login & Authentication UI
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build login page with role-based access (Employee/Client), auth context, and protected routes.

**Acceptance Criteria:**
- [x] Login page UI
- [x] Role selection (Employee / Client)
- [x] Auth context provider
- [x] Protected route wrappers
- [x] Session management

---

### SXFL-048: Settings & Notifications UI
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Build settings modal and notification panel for real-time alerts.

**Acceptance Criteria:**
- [x] Settings modal with system config
- [x] Notification panel
- [x] Toast notifications
- [x] Preferences persistence

---

### SXFL-049: Frontend API Client & React Hooks
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create type-safe API client library with error handling, loading states, and custom React hooks.

**Acceptance Criteria:**
- [x] Type-safe API client
- [x] Error handling
- [x] Loading states
- [x] Custom React hooks for data fetching

---

### SXFL-050: Cross-Browser Testing & Responsive QA
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Manual testing of all dashboard pages on Chrome, Firefox, Edge. Verify responsive behavior.

**Acceptance Criteria:**
- [x] Chrome tested
- [x] Firefox tested
- [x] Edge tested
- [x] Responsive verification
- [x] Bug report submitted

---

**Sprint 5 Total: 48 pts** (Anshuman: 3 pts, Komal: 45 pts)

---

## Sprint 6: Integration & Polish (Mar 12-19)

### SXFL-051: End-to-End Pipeline Integration
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 8 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Full integration testing: Data → PII → Models → FL → Reports → Dashboard.

**Acceptance Criteria:**
- [x] Data → PII → Models works
- [x] FL simulation completes
- [x] Reports generate
- [x] Dashboard displays all data

---

### SXFL-052: Demo Script & Presentation Pipeline
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create demo script for panel presentation covering full pipeline.

**Acceptance Criteria:**
- [x] Full pipeline demo
- [x] Error handling
- [x] Clear output
- [x] 10-minute runtime

---

### SXFL-053: Performance & Memory Optimization
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Optimize performance and memory usage across the stack.

**Acceptance Criteria:**
- [x] VRAM stays under 4GB
- [x] RAM stays under 16GB
- [x] Response times < 5s
- [x] No memory leaks

---

### SXFL-054: Error Handling & Recovery Polish
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Add comprehensive error handling across all API endpoints.

**Acceptance Criteria:**
- [x] All API errors handled
- [x] User-friendly messages
- [x] Logging complete
- [x] Recovery mechanisms

---

### SXFL-055: Documentation & User Guide
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Status** | Done |

**Description:**
Format and polish all documentation, create user guide with screenshots.

**Acceptance Criteria:**
- [x] README polished
- [x] API docs formatted
- [x] User guide created
- [x] Architecture diagrams exported

---

### SXFL-056: Presentation Slides
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create 20-30 slide panel presentation.

**Acceptance Criteria:**
- [x] 20-30 slides
- [x] Problem → Solution flow
- [x] Architecture diagram
- [x] Demo screenshots
- [x] Results summary

---

### SXFL-057: Dockerfile & Deployment Config
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Status** | Done |

**Description:**
Create Dockerfile for containerized deployment and write deployment documentation.

**Acceptance Criteria:**
- [x] Dockerfile created
- [x] Environment variables configured
- [x] Deployment docs written

---

**Sprint 6 Total: 34 pts** (Anshuman: 29 pts, Komal: 5 pts)

---

## Summary

### Story Points by Assignee

| Assignee | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | Sprint 5 | Sprint 6 | **Total** |
|----------|----------|----------|----------|----------|----------|----------|-----------|
| Anshuman | 42 | 40 | 40 | 35 | 3 | 29 | **189** (80%) |
| Komal | 0 | 0 | 0 | 0 | 45 | 5 | **50** (20%) |
| **Total** | **42** | **40** | **40** | **35** | **48** | **34** | **239** |

### Anshuman's Scope Summary

| Area | Tasks | Key Deliverables |
|------|-------|-----------------|
| Architecture | SXFL-001, 002 | Project setup, FastAPI skeleton, config management |
| Data Pipeline | SXFL-003, 004, 005 | DuckDB loader, splitters, data API |
| PII Pipeline (Patent) | SXFL-006 to 011 | 5-Gate certified sanitization, audit trail, certificates |
| ML Models | SXFL-012 to 020 | XGBoost, LightGBM, TabNet, Ensemble, SHAP, MLflow |
| Differential Privacy | SXFL-021 to 024 | Gaussian mechanism, RDP accountant, budget manager |
| Federated Learning | SXFL-025 to 030 | Flower server/client, Multi-Krum, FL simulator |
| LLM & RAG | SXFL-031 to 039 | Phi-3, ChromaDB, RAG, hallucination guards, reports |
| Integration | SXFL-049, 051-054, 056-057 | API client, E2E pipeline, optimization, deployment |

### Komal's Scope Summary

| Area | Tasks | Key Deliverables |
|------|-------|-----------------|
| Dashboard Setup | SXFL-040 | React + Vite + Tailwind project initialization |
| Dashboard Layout | SXFL-041 | Sidebar, header, navigation, responsive layout |
| Dashboard Pages | SXFL-042 to 046 | Executive Overview, Transactions, FL Status, Privacy, Explainability |
| Auth & Settings UI | SXFL-047, 048 | Login page, role-based auth UI, settings modal, notifications |
| QA & Docs | SXFL-050, 055 | Cross-browser testing, documentation formatting |

### Patent-Critical Tasks (Anshuman Only)

| Task IDs | Feature | Patent Claim |
|----------|---------|--------------|
| SXFL-006 to 011 | 5-Gate PII Pipeline | Claim 1: Certified Data Sanitization |
| SXFL-027, 028 | Byzantine Robustness | Claim 2: Byzantine-Robust FL |
| SXFL-021 to 023 | DP Mechanisms | Claim 3: Composable Privacy Budget |
| SXFL-036, 037 | Hallucination Guards | Claim 4: Grounded XAI |
| SXFL-013 to 017 | Multi-Model Framework | Claim 5: Multi-Fraud Detection |

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 20, 2026*
