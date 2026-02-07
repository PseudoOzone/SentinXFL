# SentinXFL - Task Breakdown (JIRA Style)

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Project Lead**: Anshuman Bakshi (RA2211033010117)  
> **Contributor**: Komal (RA2211033010114)

---

## Sprint Overview

| Sprint | Dates | Story Points | Focus |
|--------|-------|--------------|-------|
| Sprint 1 | Feb 5-12 | 34 pts | Foundation + PII |
| Sprint 2 | Feb 12-19 | 32 pts | ML Models |
| Sprint 3 | Feb 19-26 | 38 pts | FL + DP |
| Sprint 4 | Feb 26 - Mar 5 | 30 pts | LLM + RAG |
| Sprint 5 | Mar 5-12 | 34 pts | Dashboard |
| Sprint 6 | Mar 12-19 | 30 pts | Integration |
| **Total** | - | **198 pts** | - |

---

## Team Allocation

```
┌─────────────────────────────────────────────────────────────────┐
│                    TEAM ALLOCATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Anshuman Bakshi (Lead)                                         │
│  ├── ALL Core Development                172 pts (87%)          │
│  ├── ALL Algorithm Implementation                               │
│  ├── ALL API Development                                        │
│  ├── ALL ML/FL/DP Code                                          │
│  ├── ALL Patent-Critical Features                               │
│  └── System Architecture                                        │
│                                                                 │
│  Komal (Contributor)                                            │
│  ├── UI Research & References              8 pts (4%)           │
│  ├── Documentation Formatting              8 pts (4%)           │
│  ├── Manual Testing                        6 pts (3%)           │
│  └── Presentation Support                  4 pts (2%)           │
│                                                     26 pts (13%)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Epic Breakdown

### EPIC-1: Data Infrastructure
**Owner**: Anshuman | **Points**: 21 | **Sprint**: 1

### EPIC-2: PII Blocking Pipeline (PATENT CORE)
**Owner**: Anshuman | **Points**: 26 | **Sprint**: 1

### EPIC-3: ML Model Framework
**Owner**: Anshuman | **Points**: 32 | **Sprint**: 2

### EPIC-4: Federated Learning
**Owner**: Anshuman | **Points**: 24 | **Sprint**: 3

### EPIC-5: Differential Privacy
**Owner**: Anshuman | **Points**: 14 | **Sprint**: 3

### EPIC-6: LLM & RAG
**Owner**: Anshuman | **Points**: 30 | **Sprint**: 4

### EPIC-7: Dashboard
**Owner**: Anshuman | **Points**: 34 | **Sprint**: 5

### EPIC-8: Documentation & Testing
**Owner**: Komal (research) + Anshuman (implementation) | **Points**: 17 | **Sprint**: 6

---

## Sprint 1: Foundation + PII (Feb 5-12)

### SXFL-001: Project Setup
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Initialize project structure with proper directory layout, git repository, and dependency management.

**Acceptance Criteria:**
- [ ] Project structure created
- [ ] pyproject.toml with all dependencies
- [ ] .gitignore configured
- [ ] Pre-commit hooks set up
- [ ] Git initialized with initial commit

---

### SXFL-002: FastAPI Skeleton
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Create FastAPI application skeleton with configuration management, logging, and health endpoint.

**Acceptance Criteria:**
- [ ] FastAPI app runs on localhost:8000
- [ ] Health check endpoint returns 200
- [ ] Configuration via environment variables
- [ ] Structured logging configured

---

### SXFL-003: DuckDB Loader
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement lazy CSV loader using DuckDB for all three datasets.

**Acceptance Criteria:**
- [ ] Bank Account Fraud dataset loads (6M rows)
- [ ] Credit Card dataset loads (285K rows)
- [ ] PaySim dataset loads (6.3M rows)
- [ ] Schema detection working
- [ ] 10+ unit tests passing

---

### SXFL-004: Data Splitter
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement temporal and stratified splitters for train/val/test splits.

**Acceptance Criteria:**
- [ ] Temporal split working
- [ ] Stratified split working
- [ ] No data leakage validation
- [ ] Configurable ratios

---

### SXFL-005: Data API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 2 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Create REST API endpoints for data operations.

**Acceptance Criteria:**
- [ ] GET /api/v1/data/datasets
- [ ] GET /api/v1/data/{dataset}/schema
- [ ] POST /api/v1/data/{dataset}/split
- [ ] Integration tests passing

---

### SXFL-006: Statistical PII Detection
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement entropy and cardinality-based PII detection.

**Acceptance Criteria:**
- [ ] Entropy calculation working
- [ ] Cardinality ratio calculation
- [ ] PII columns auto-detected
- [ ] Confidence scores assigned
- [ ] 15+ unit tests

---

### SXFL-007: Pattern-Based PII Detection
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement regex pattern matching for known PII formats.

**Acceptance Criteria:**
- [ ] Credit card pattern
- [ ] SSN pattern
- [ ] Email pattern
- [ ] Phone pattern
- [ ] Indian ID patterns (Aadhaar, PAN)

---

### SXFL-008: k-Anonymity Analyzer
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement k-anonymity verification and re-identification risk scoring.

**Acceptance Criteria:**
- [ ] k-anonymity calculation
- [ ] Quasi-identifier combination analysis
- [ ] Risk score (1/k)
- [ ] Configurable k threshold

---

### SXFL-009: PII Transformations
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement utility-preserving PII transformations.

**Acceptance Criteria:**
- [ ] Binning transform
- [ ] Generalization transform
- [ ] Suppression transform
- [ ] DP noise transform
- [ ] Hard blocking gate

---

### SXFL-010: PII Certificate Generator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Create certificate generation with schema hashing.

**Acceptance Criteria:**
- [ ] Certificate generation
- [ ] Schema hash verification
- [ ] Certificate storage
- [ ] Verification endpoint

---

### SXFL-011: Audit Trail
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 1 |

**Description:**
Implement hash-chained audit log with tamper detection.

**Acceptance Criteria:**
- [ ] Hash-chained entries
- [ ] Tamper detection
- [ ] Audit API endpoints
- [ ] Query by time range

---

### SXFL-012: UI Reference Research
| Field | Value |
|-------|-------|
| **Type** | Research |
| **Priority** | Low |
| **Story Points** | 3 |
| **Assignee** | Komal |
| **Sprint** | 1 |

**Description:**
Research dark theme dashboard designs and collect UI references.

**Acceptance Criteria:**
- [ ] 10+ reference screenshots collected
- [ ] Color palette documented
- [ ] Component examples noted
- [ ] Shared with Anshuman

---

**Sprint 1 Total: 45 pts** (Anshuman: 42 pts, Komal: 3 pts)

---

## Sprint 2: ML Models (Feb 12-19)

### SXFL-013: Base Model Interface
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Create abstract base class for all models with common interface.

**Acceptance Criteria:**
- [ ] fit() interface
- [ ] predict() interface
- [ ] predict_proba() interface
- [ ] save/load interface
- [ ] Metrics calculator

---

### SXFL-014: XGBoost Implementation
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Implement XGBoost wrapper with hyperparameter configuration.

**Acceptance Criteria:**
- [ ] XGBoost training works
- [ ] Early stopping configured
- [ ] Hyperparameter config
- [ ] AUC-ROC > 0.90 on all datasets
- [ ] Serialization working

---

### SXFL-015: LightGBM Implementation
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Implement LightGBM wrapper optimized for CPU.

**Acceptance Criteria:**
- [ ] LightGBM training works
- [ ] CPU-optimized configuration
- [ ] Comparable to XGBoost
- [ ] Unit tests passing

---

### SXFL-016: IsolationForest Implementation
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Implement IsolationForest for anomaly-based detection.

**Acceptance Criteria:**
- [ ] IsolationForest training works
- [ ] Anomaly scores → probabilities
- [ ] Contamination tuned
- [ ] Unit tests passing

---

### SXFL-017: TabNet Implementation
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 8 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Implement TabNet optimized for 4GB VRAM.

**Acceptance Criteria:**
- [ ] TabNet trains without OOM
- [ ] VRAM usage ≤ 1GB
- [ ] Attention weights extractable
- [ ] Performance comparable
- [ ] GPU memory management

---

### SXFL-018: Ensemble Model
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Implement weighted ensemble combining all models.

**Acceptance Criteria:**
- [ ] Weighted averaging
- [ ] Weight optimization
- [ ] Probability calibration
- [ ] Stacking option

---

### SXFL-019: SHAP Integration
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Integrate SHAP for model explanations.

**Acceptance Criteria:**
- [ ] SHAP values computed
- [ ] Feature importance ranked
- [ ] Explanation API
- [ ] Caching for speed

---

### SXFL-020: MLflow Tracking
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Set up MLflow for experiment tracking.

**Acceptance Criteria:**
- [ ] Metrics logged
- [ ] Model artifacts stored
- [ ] Experiment management
- [ ] UI accessible

---

### SXFL-021: Model API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 2 |

**Description:**
Create REST API endpoints for model operations.

**Acceptance Criteria:**
- [ ] POST /api/v1/models/train
- [ ] POST /api/v1/models/predict
- [ ] GET /api/v1/models/{id}/metrics
- [ ] GET /api/v1/models/{id}/explain

---

**Sprint 2 Total: 40 pts** (Anshuman: 40 pts, Komal: 0 pts)

---

## Sprint 3: FL + DP (Feb 19-26)

### SXFL-022: Gaussian Mechanism
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Implement Gaussian mechanism for differential privacy.

**Acceptance Criteria:**
- [ ] Noise correctly calibrated
- [ ] (ε,δ) guarantees
- [ ] Configurable parameters
- [ ] Unit tests

---

### SXFL-023: RDP Accountant
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Implement Rényi Differential Privacy accountant.

**Acceptance Criteria:**
- [ ] RDP composition
- [ ] (ε,δ) converter
- [ ] Tight bounds
- [ ] Unit tests

---

### SXFL-024: Privacy Budget Manager
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Track and enforce privacy budget across operations.

**Acceptance Criteria:**
- [ ] Budget tracking per client
- [ ] Aggregate budget tracking
- [ ] Budget enforcement
- [ ] Query API

---

### SXFL-025: DP API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Create REST API for DP operations.

**Acceptance Criteria:**
- [ ] GET /api/v1/dp/budget
- [ ] POST /api/v1/dp/query
- [ ] GET /api/v1/dp/history

---

### SXFL-026: Flower Server
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Implement Flower server with FedAvg strategy.

**Acceptance Criteria:**
- [ ] Server starts
- [ ] FedAvg aggregation
- [ ] Round management
- [ ] Client tracking

---

### SXFL-027: Flower Client
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Implement Flower client with local training.

**Acceptance Criteria:**
- [ ] Client connects to server
- [ ] Local training works
- [ ] Gradient extraction
- [ ] DP integration

---

### SXFL-028: Multi-Krum Aggregator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Implement Byzantine-robust Multi-Krum aggregation.

**Acceptance Criteria:**
- [ ] Multi-Krum implemented
- [ ] Tolerates f malicious
- [ ] Configurable f
- [ ] Unit tests

---

### SXFL-029: Trimmed Mean & Median
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Implement Trimmed Mean and Coordinate Median.

**Acceptance Criteria:**
- [ ] Trimmed Mean implemented
- [ ] Coordinate Median implemented
- [ ] Configurable trimming ratio
- [ ] Unit tests

---

### SXFL-030: FL Simulator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Single-machine FL simulation for testing.

**Acceptance Criteria:**
- [ ] Simulate 3+ clients
- [ ] Data partitioning
- [ ] FL converges
- [ ] Metrics tracked

---

### SXFL-031: FL API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 3 |

**Description:**
Create REST API for FL operations.

**Acceptance Criteria:**
- [ ] POST /api/v1/fl/simulate
- [ ] GET /api/v1/fl/status
- [ ] GET /api/v1/fl/history

---

**Sprint 3 Total: 40 pts** (Anshuman: 40 pts, Komal: 0 pts)

---

## Sprint 4: LLM + RAG (Feb 26 - Mar 5)

### SXFL-032: Phi-3 Model Loading
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Load Phi-3-mini-4k with 4-bit quantization.

**Acceptance Criteria:**
- [ ] Model loads
- [ ] VRAM ≤ 2GB
- [ ] Basic generation works
- [ ] Memory management

---

### SXFL-033: Prompt Templates
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Create prompt templates for report generation.

**Acceptance Criteria:**
- [ ] Executive summary template
- [ ] Evidence report template
- [ ] Technical report template
- [ ] Template rendering

---

### SXFL-034: ChromaDB Setup
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Set up ChromaDB vector store.

**Acceptance Criteria:**
- [ ] ChromaDB running
- [ ] Collections created
- [ ] Persistence working
- [ ] Unit tests

---

### SXFL-035: Embedding Pipeline
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Create embedding generation pipeline.

**Acceptance Criteria:**
- [ ] MiniLM embeddings
- [ ] Batch processing
- [ ] ChromaDB integration
- [ ] Unit tests

---

### SXFL-036: RAG Retriever
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Implement retrieval with MMR diversity.

**Acceptance Criteria:**
- [ ] Semantic retrieval
- [ ] MMR diversity
- [ ] Relevance filtering
- [ ] Configurable k

---

### SXFL-037: Evidence Collector
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Collect SHAP and TabNet evidence for reports.

**Acceptance Criteria:**
- [ ] SHAP values collected
- [ ] TabNet attention collected
- [ ] Evidence formatted
- [ ] Injected into prompts

---

### SXFL-038: Hallucination Guards
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Implement NLI-based hallucination detection.

**Acceptance Criteria:**
- [ ] Claim extraction
- [ ] Entailment checking
- [ ] Confidence scoring
- [ ] Warnings generated

---

### SXFL-039: Report Generator
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Generate full fraud reports with LLM.

**Acceptance Criteria:**
- [ ] Executive summary generated
- [ ] Evidence integrated
- [ ] PDF export
- [ ] Quality validation

---

### SXFL-040: Report API Endpoints
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 4 |

**Description:**
Create REST API for report operations.

**Acceptance Criteria:**
- [ ] POST /api/v1/reports/generate
- [ ] GET /api/v1/reports/{id}
- [ ] GET /api/v1/reports/history

---

**Sprint 4 Total: 35 pts** (Anshuman: 35 pts, Komal: 0 pts)

---

## Sprint 5: Dashboard (Mar 5-12)

### SXFL-041: Next.js Setup
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Critical |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Initialize Next.js 14 with Tailwind and shadcn/ui.

**Acceptance Criteria:**
- [ ] Next.js running
- [ ] Tailwind configured
- [ ] shadcn/ui installed
- [ ] Dark theme applied

---

### SXFL-042: Layout Components
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Create layout with sidebar, header, footer.

**Acceptance Criteria:**
- [ ] Main layout complete
- [ ] Sidebar navigation
- [ ] Header with logo
- [ ] Footer

---

### SXFL-043: Executive Overview Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 8 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Build Executive Overview dashboard page.

**Acceptance Criteria:**
- [ ] KPI cards
- [ ] Risk score gauge
- [ ] Risk distribution donut
- [ ] Timeline chart
- [ ] Recent changes list
- [ ] API integration

---

### SXFL-044: Local Bank View Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Build Local Bank View page.

**Acceptance Criteria:**
- [ ] Model comparison table
- [ ] Performance bar chart
- [ ] Confusion matrix
- [ ] Feature importance
- [ ] Training history

---

### SXFL-045: Central Knowledge Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Build Central Knowledge page.

**Acceptance Criteria:**
- [ ] Executive summary card
- [ ] Emerging patterns list
- [ ] Fraud timeline
- [ ] Bank selector
- [ ] Recommended actions

---

### SXFL-046: Technical Appendix Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Build Technical Appendix page.

**Acceptance Criteria:**
- [ ] Privacy budget tracker
- [ ] FL status panel
- [ ] Node status list
- [ ] PII audit display
- [ ] Settings panel

---

### SXFL-047: Export Page
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Build Export page.

**Acceptance Criteria:**
- [ ] Report form
- [ ] Date range picker
- [ ] Report type selector
- [ ] Report history
- [ ] Download buttons

---

### SXFL-048: API Client Library
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 5 |

**Description:**
Create frontend API client for backend.

**Acceptance Criteria:**
- [ ] Type-safe API client
- [ ] Error handling
- [ ] Loading states
- [ ] Caching

---

### SXFL-049: Dashboard Manual Testing
| Field | Value |
|-------|-------|
| **Type** | Testing |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Komal |
| **Sprint** | 5 |

**Description:**
Manual testing of dashboard on different browsers.

**Acceptance Criteria:**
- [ ] Chrome tested
- [ ] Firefox tested
- [ ] Edge tested
- [ ] Bug report submitted

---

**Sprint 5 Total: 38 pts** (Anshuman: 35 pts, Komal: 3 pts)

---

## Sprint 6: Integration (Mar 12-19)

### SXFL-050: End-to-End Pipeline
| Field | Value |
|-------|-------|
| **Type** | Story |
| **Priority** | Critical |
| **Story Points** | 8 |
| **Assignee** | Anshuman |
| **Sprint** | 6 |

**Description:**
Full integration testing of complete pipeline.

**Acceptance Criteria:**
- [ ] Data → PII → Models works
- [ ] FL simulation completes
- [ ] Reports generate
- [ ] Dashboard displays all data

---

### SXFL-051: Demo Script
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 6 |

**Description:**
Create demo script for panel presentation.

**Acceptance Criteria:**
- [ ] Full pipeline demo
- [ ] Error handling
- [ ] Clear output
- [ ] 10-minute runtime

---

### SXFL-052: Performance Optimization
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 6 |

**Description:**
Optimize performance and memory usage.

**Acceptance Criteria:**
- [ ] VRAM stays under 4GB
- [ ] RAM stays under 16GB
- [ ] Response times < 5s
- [ ] No memory leaks

---

### SXFL-053: Error Handling Polish
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | Medium |
| **Story Points** | 3 |
| **Assignee** | Anshuman |
| **Sprint** | 6 |

**Description:**
Add comprehensive error handling.

**Acceptance Criteria:**
- [ ] All API errors handled
- [ ] User-friendly messages
- [ ] Logging complete
- [ ] Recovery mechanisms

---

### SXFL-054: Documentation Formatting
| Field | Value |
|-------|-------|
| **Type** | Documentation |
| **Priority** | Medium |
| **Story Points** | 5 |
| **Assignee** | Komal |
| **Sprint** | 6 |

**Description:**
Format and polish all documentation.

**Acceptance Criteria:**
- [ ] README polished
- [ ] API docs formatted
- [ ] User guide created
- [ ] Architecture diagrams exported

---

### SXFL-055: Presentation Slides
| Field | Value |
|-------|-------|
| **Type** | Task |
| **Priority** | High |
| **Story Points** | 5 |
| **Assignee** | Anshuman |
| **Sprint** | 6 |

**Description:**
Create panel presentation slides.

**Acceptance Criteria:**
- [ ] 20-30 slides
- [ ] Problem → Solution flow
- [ ] Architecture diagram
- [ ] Demo screenshots
- [ ] Results summary

---

### SXFL-056: Presentation Research Support
| Field | Value |
|-------|-------|
| **Type** | Research |
| **Priority** | Low |
| **Story Points** | 4 |
| **Assignee** | Komal |
| **Sprint** | 6 |

**Description:**
Research presentation templates and assist with formatting.

**Acceptance Criteria:**
- [ ] Template suggestions
- [ ] Color scheme recommendations
- [ ] Font suggestions
- [ ] Review feedback

---

**Sprint 6 Total: 35 pts** (Anshuman: 26 pts, Komal: 9 pts)

---

## Summary

### Story Points by Assignee

| Assignee | Sprint 1 | Sprint 2 | Sprint 3 | Sprint 4 | Sprint 5 | Sprint 6 | **Total** |
|----------|----------|----------|----------|----------|----------|----------|-----------|
| Anshuman | 42 | 40 | 40 | 35 | 35 | 26 | **218** (87%) |
| Komal | 3 | 0 | 0 | 0 | 3 | 9 | **15** (13%) |
| **Total** | **45** | **40** | **40** | **35** | **38** | **35** | **233** |

### Komal's Tasks Summary

| Task ID | Title | Points | Sprint | Type |
|---------|-------|--------|--------|------|
| SXFL-012 | UI Reference Research | 3 | 1 | Research |
| SXFL-049 | Dashboard Manual Testing | 3 | 5 | Testing |
| SXFL-054 | Documentation Formatting | 5 | 6 | Documentation |
| SXFL-056 | Presentation Research Support | 4 | 6 | Research |
| **Total** | - | **15** | - | - |

### Patent-Critical Tasks (Anshuman Only)

| Task ID | Feature | Patent Claim |
|---------|---------|--------------|
| SXFL-006 to SXFL-011 | PII Pipeline | Claim 1: Certified Data Sanitization |
| SXFL-028 to SXFL-029 | Byzantine Robustness | Claim 2: Byzantine-Robust FL |
| SXFL-022 to SXFL-024 | DP Mechanisms | Claim 3: Composable Privacy Budget |
| SXFL-037 to SXFL-038 | Hallucination Guards | Claim 4: Grounded XAI |
| SXFL-014 to SXFL-018 | Multi-Model Framework | Claim 5: Multi-Fraud Detection |

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
