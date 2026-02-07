# SentinXFL - Implementation Plan

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)  
> **Timeline**: February 5, 2026 → May 1, 2026 (Panel Presentation)

---

## 1. Timeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT TIMELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Feb 5                                                          May 1      │
│    │                                                              │        │
│    ▼                                                              ▼        │
│    ├────────┬────────┬────────┬────────┬────────┬────────────────┤        │
│    │ Week 1 │ Week 2 │ Week 3 │ Week 4 │ Week 5 │  Buffer (6w)   │        │
│    │ Found- │ ML     │ FL +   │ LLM +  │ Dash-  │  Polish +      │        │
│    │ ation  │ Models │ DP     │ RAG    │ board  │  Publication   │        │
│    ├────────┴────────┴────────┴────────┴────────┴────────────────┤        │
│    │                                                              │        │
│    │  Feb 12   Feb 19   Feb 26   Mar 5   Mar 12            May 1 │        │
│    │                                                              │        │
│    │  Checkpoints:                                                │        │
│    │  ✓ Week 2: Models working                                    │        │
│    │  ✓ Week 3: FL simulation complete                            │        │
│    │  ✓ Week 4: Reports generating                                │        │
│    │  ✓ Week 5: Dashboard complete                                │        │
│    │  ✓ Mar 19: MVP READY                                         │        │
│    │  ✓ Apr 15: Paper submitted                                   │        │
│    │  ✓ May 1: Panel presentation                                 │        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 1: Foundation (Feb 5-12)

### 2.1 Day 1 (Feb 5): Project Setup

**Morning: Project Structure**
```
Tasks:
├── Create project directory structure
├── Initialize git repository
├── Create pyproject.toml with all dependencies
├── Create .gitignore, .env.example
├── Set up pre-commit hooks
└── Create knowledge/ folder with docs

Deliverables:
✓ Project structure created
✓ Dependencies defined
✓ Git initialized
```

**Afternoon: Backend Skeleton**
```
Tasks:
├── Create FastAPI app skeleton
├── Set up configuration management (Pydantic settings)
├── Create logging setup
├── Create health check endpoint
└── Verify uvicorn runs

Deliverables:
✓ FastAPI app running on localhost:8000
✓ /health endpoint returning OK
```

### 2.2 Day 2 (Feb 6): Data Pipeline

**Morning: DuckDB Loader**
```
Tasks:
├── Implement DuckDB connection manager
├── Create lazy CSV loader
├── Implement schema detection
├── Create dataset registry
└── Write unit tests

Code Focus:
- src/sentinxfl/data/loader.py
- src/sentinxfl/data/schema.py

Deliverables:
✓ Load Bank Account Fraud dataset (1M rows sample)
✓ Load Credit Card dataset
✓ Load PaySim dataset
✓ 10+ unit tests passing
```

**Afternoon: Data Splits**
```
Tasks:
├── Implement temporal splitter
├── Implement stratified splitter
├── Create split validation (no leakage)
├── Add API endpoints for data operations
└── Write integration tests

Code Focus:
- src/sentinxfl/data/splitter.py
- src/sentinxfl/api/routes/data.py

Deliverables:
✓ Temporal split working
✓ Train/val/test ratios correct
✓ API endpoints functional
```

### 2.3 Day 3 (Feb 7): PII Detection

**Morning: Statistical Detection**
```
Tasks:
├── Implement entropy calculator
├── Implement cardinality analyzer
├── Create statistical PII detector
├── Add confidence scoring
└── Write tests

Code Focus:
- src/sentinxfl/pii/detector.py

Algorithms:
- Entropy: H(X) = -Σ p(x) log p(x)
- Cardinality ratio: unique/total
- Thresholds: entropy > 4.0, cardinality > 0.9

Deliverables:
✓ Detect PII columns automatically
✓ Confidence scores assigned
✓ 15+ unit tests
```

**Afternoon: Pattern Matching**
```
Tasks:
├── Implement regex patterns (CC, SSN, email, phone)
├── Create pattern matcher
├── Integrate with detector
├── Add PII API endpoints
└── Test on real datasets

Code Focus:
- src/sentinxfl/pii/patterns.py
- src/sentinxfl/api/routes/pii.py

Deliverables:
✓ Pattern detection working
✓ API endpoints for detection
```

### 2.4 Day 4 (Feb 8): PII Blocking & Transforms

**Morning: Quasi-Identifier Analysis**
```
Tasks:
├── Implement k-anonymity checker
├── Create quasi-identifier combination analyzer
├── Calculate re-identification risk
├── Add risk scoring
└── Write tests

Code Focus:
- src/sentinxfl/pii/analyzer.py

Algorithms:
- k-anonymity: Count rows with identical quasi-ID combinations
- Risk: 1/k for smallest equivalence class

Deliverables:
✓ k-anonymity verification
✓ Risk score calculation
```

**Afternoon: Transformations**
```
Tasks:
├── Implement binning transform
├── Implement generalization transform
├── Implement suppression transform
├── Implement DP noise transform
├── Create hard blocking gate
└── Write tests

Code Focus:
- src/sentinxfl/pii/transformer.py
- src/sentinxfl/pii/blocker.py

Deliverables:
✓ All transforms working
✓ Blocking gate enforced
✓ No PII passes through
```

### 2.5 Day 5 (Feb 9): PII Certification & Audit

**Morning: Certification System**
```
Tasks:
├── Create certificate generator
├── Implement schema hashing
├── Create certificate storage
├── Add verification endpoint
└── Write tests

Code Focus:
- src/sentinxfl/pii/certifier.py

Deliverables:
✓ Certificates generated
✓ Schema hash verification
✓ API endpoints working
```

**Afternoon: Audit Trail**
```
Tasks:
├── Implement hash-chained audit log
├── Create audit entry model
├── Add tamper detection
├── Create audit API
└── Integration testing

Code Focus:
- src/sentinxfl/pii/audit.py

Deliverables:
✓ Audit trail complete
✓ Tamper-proof logging
✓ Full PII pipeline working end-to-end
```

### 2.6 Days 6-7 (Feb 10-11): Testing & Documentation

**Weekend Tasks**
```
Tasks:
├── Write comprehensive unit tests (50+ tests)
├── Create integration tests
├── Document PII pipeline
├── Create API documentation
├── Performance testing
└── Bug fixes

Deliverables:
✓ 80%+ test coverage on PII module
✓ API docs generated
✓ Phase 1 complete
```

---

## 3. Phase 2: ML Models (Feb 12-19)

### 3.1 Day 8 (Feb 12): Base Interface & XGBoost

**Morning: Base Model Interface**
```
Tasks:
├── Create abstract base model class
├── Define fit/predict/predict_proba interface
├── Define save/load interface
├── Create metrics calculator
└── Write tests

Code Focus:
- src/sentinxfl/models/base.py
- src/sentinxfl/models/metrics.py

Deliverables:
✓ Abstract interface defined
✓ Metrics calculator working
```

**Afternoon: XGBoost Implementation**
```
Tasks:
├── Implement XGBoost wrapper
├── Add hyperparameter configuration
├── Implement training with early stopping
├── Add serialization
└── Test on all datasets

Code Focus:
- src/sentinxfl/models/xgboost_model.py

Deliverables:
✓ XGBoost training on all datasets
✓ AUC-ROC > 0.90
```

### 3.2 Day 9 (Feb 13): LightGBM & IsolationForest

**Morning: LightGBM**
```
Tasks:
├── Implement LightGBM wrapper
├── Configure for CPU efficiency
├── Add training logic
├── Compare with XGBoost
└── Write tests

Code Focus:
- src/sentinxfl/models/lightgbm_model.py

Deliverables:
✓ LightGBM working
✓ Comparable to XGBoost
```

**Afternoon: IsolationForest**
```
Tasks:
├── Implement IsolationForest wrapper
├── Handle predict_proba (anomaly scores)
├── Tune contamination parameter
├── Add to model registry
└── Write tests

Code Focus:
- src/sentinxfl/models/isolation_model.py

Deliverables:
✓ IsolationForest working
✓ Anomaly scores converted to probabilities
```

### 3.3 Day 10 (Feb 14): TabNet

**Morning: TabNet Implementation**
```
Tasks:
├── Implement TabNet wrapper
├── Configure for 4GB VRAM
├── Add GPU memory management
├── Implement attention extraction
└── Test on sample data

Code Focus:
- src/sentinxfl/models/tabnet_model.py

VRAM Optimization:
- batch_size: 1024
- n_d, n_a: 8 (reduced)
- Clear cache between runs

Deliverables:
✓ TabNet training without OOM
✓ ~1GB VRAM usage
```

**Afternoon: TabNet Testing**
```
Tasks:
├── Train on full datasets
├── Extract attention weights
├── Compare performance
├── Memory profiling
└── Bug fixes

Deliverables:
✓ TabNet comparable to boosting
✓ Attention weights accessible
```

### 3.4 Day 11 (Feb 15): Ensemble

**Morning: Ensemble Implementation**
```
Tasks:
├── Implement weighted ensemble
├── Add weight optimization
├── Create stacking option
├── Add calibration (Platt scaling)
└── Write tests

Code Focus:
- src/sentinxfl/models/ensemble.py

Deliverables:
✓ Ensemble combines all models
✓ Weights optimized
✓ Calibrated probabilities
```

**Afternoon: Model API**
```
Tasks:
├── Create model API endpoints
├── Add training endpoint
├── Add prediction endpoint
├── Add metrics endpoint
├── Integration testing

Code Focus:
- src/sentinxfl/api/routes/models.py

Deliverables:
✓ All model APIs working
✓ Can train/predict via API
```

### 3.5 Day 12 (Feb 16): SHAP & MLflow

**Morning: SHAP Explanations**
```
Tasks:
├── Integrate SHAP with models
├── Create explanation generator
├── Add feature importance
├── Optimize for speed
└── Write tests

Code Focus:
- src/sentinxfl/models/explain.py

Deliverables:
✓ SHAP values computed
✓ Feature importance ranked
```

**Afternoon: MLflow Integration**
```
Tasks:
├── Set up MLflow tracking
├── Log training metrics
├── Log model artifacts
├── Create experiment management
└── Test tracking

Code Focus:
- src/sentinxfl/models/tracking.py

Deliverables:
✓ MLflow tracking working
✓ Experiments logged
```

### 3.6 Days 13-14 (Feb 17-18): Testing & Benchmarks

**Weekend Tasks**
```
Tasks:
├── Comprehensive model testing
├── Performance benchmarks
├── Memory profiling
├── Compare all models
├── Documentation
└── Bug fixes

Deliverables:
✓ All models tested
✓ Benchmark results documented
✓ Phase 2 complete
```

---

## 4. Phase 3: FL + DP (Feb 19-26)

### 4.1 Day 15 (Feb 19): Differential Privacy

**Morning: DP Mechanisms**
```
Tasks:
├── Implement Gaussian mechanism
├── Implement Laplace mechanism
├── Add gradient clipping
├── Create noise calibration
└── Write tests

Code Focus:
- src/sentinxfl/dp/mechanisms.py

Deliverables:
✓ DP mechanisms working
✓ Noise correctly calibrated
```

**Afternoon: RDP Accountant**
```
Tasks:
├── Implement RDP accountant
├── Add composition rules
├── Create (ε,δ) converter
├── Add budget tracking
└── Write tests

Code Focus:
- src/sentinxfl/dp/accountant.py
- src/sentinxfl/dp/budget.py

Deliverables:
✓ Budget tracking working
✓ RDP composition tight
```

### 4.2 Day 16 (Feb 20): DP Integration

**Morning: DP API**
```
Tasks:
├── Create DP API endpoints
├── Add budget query endpoint
├── Add DP query endpoint
├── Implement budget enforcement
└── Write tests

Code Focus:
- src/sentinxfl/api/routes/dp.py

Deliverables:
✓ DP APIs working
✓ Budget enforcement active
```

**Afternoon: DP Release**
```
Tasks:
├── Create DP data release
├── Add DP statistics
├── Integrate with models
├── Test privacy guarantees
└── Documentation

Code Focus:
- src/sentinxfl/dp/release.py

Deliverables:
✓ DP release working
✓ Privacy verified
```

### 4.3 Day 17 (Feb 21): Flower Server

**Morning: FL Server Setup**
```
Tasks:
├── Create Flower server
├── Implement FedAvg strategy
├── Add round management
├── Create server API
└── Write tests

Code Focus:
- src/sentinxfl/fl/server.py
- src/sentinxfl/fl/strategy.py

Deliverables:
✓ Flower server running
✓ FedAvg working
```

**Afternoon: FL Client**
```
Tasks:
├── Create Flower client
├── Implement local training
├── Add gradient extraction
├── Integrate with DP
└── Write tests

Code Focus:
- src/sentinxfl/fl/client.py

Deliverables:
✓ Client connects to server
✓ Local training works
```

### 4.4 Day 18 (Feb 22): Byzantine Robustness

**Morning: Krum & Trimmed Mean**
```
Tasks:
├── Implement Multi-Krum
├── Implement Trimmed Mean
├── Add strategy selection
├── Write tests
└── Compare with FedAvg

Code Focus:
- src/sentinxfl/fl/aggregators/krum.py
- src/sentinxfl/fl/aggregators/trimmed_mean.py

Deliverables:
✓ Byzantine-robust aggregation
✓ Tolerates f malicious clients
```

**Afternoon: Coordinate Median**
```
Tasks:
├── Implement Coordinate Median
├── Integrate all aggregators
├── Create aggregator factory
├── Test robustness
└── Documentation

Code Focus:
- src/sentinxfl/fl/aggregators/median.py

Deliverables:
✓ All aggregators working
✓ Robustness verified
```

### 4.5 Day 19 (Feb 23): FL Simulator

**Morning: Single-Machine Simulation**
```
Tasks:
├── Create FL simulator
├── Split data for multiple "clients"
├── Run simulated FL rounds
├── Track convergence
└── Write tests

Code Focus:
- src/sentinxfl/fl/simulator.py

Deliverables:
✓ Simulate 3+ clients
✓ FL converges
```

**Afternoon: FL API**
```
Tasks:
├── Create FL API endpoints
├── Add simulation endpoint
├── Add status endpoint
├── Integration testing
└── Bug fixes

Code Focus:
- src/sentinxfl/api/routes/fl.py

Deliverables:
✓ FL APIs working
✓ Can run simulation via API
```

### 4.6 Days 20-21 (Feb 24-25): Testing & Integration

**Weekend Tasks**
```
Tasks:
├── Full FL + DP integration testing
├── Privacy budget verification
├── Performance benchmarks
├── Documentation
├── Bug fixes
└── Phase 3 complete

Deliverables:
✓ FL + DP working together
✓ Privacy budget tracked across rounds
✓ Phase 3 complete
```

---

## 5. Phase 4: LLM + RAG (Feb 26 - Mar 5)

### 5.1 Day 22 (Feb 26): Phi-3 Setup

**Morning: Model Loading**
```
Tasks:
├── Download Phi-3-mini model
├── Configure 4-bit quantization
├── Test model loading
├── Verify VRAM usage (~2GB)
└── Basic generation test

Code Focus:
- src/sentinxfl/llm/model.py

CRITICAL: Clear TabNet from VRAM first!

Deliverables:
✓ Phi-3 loads without OOM
✓ Basic generation works
```

**Afternoon: Generation Pipeline**
```
Tasks:
├── Create generation config
├── Implement prompt templates
├── Add temperature/top-p control
├── Test various prompts
└── Memory management

Code Focus:
- src/sentinxfl/llm/templates.py

Deliverables:
✓ Generation pipeline working
✓ Templates defined
```

### 5.2 Day 23 (Feb 27): ChromaDB & Embeddings

**Morning: Vector Store**
```
Tasks:
├── Set up ChromaDB
├── Create collection management
├── Add persistence
├── Write tests
└── Verify storage

Code Focus:
- src/sentinxfl/rag/store.py

Deliverables:
✓ ChromaDB working
✓ Collections persist
```

**Afternoon: Embeddings**
```
Tasks:
├── Load MiniLM embedding model
├── Create embedding generator
├── Add batch processing
├── Integrate with ChromaDB
└── Write tests

Code Focus:
- src/sentinxfl/rag/embeddings.py

Deliverables:
✓ Embeddings generated
✓ Stored in ChromaDB
```

### 5.3 Day 24 (Feb 28): RAG Pipeline

**Morning: Retrieval**
```
Tasks:
├── Implement retrieval logic
├── Add MMR for diversity
├── Create relevance filtering
├── Optimize retrieval speed
└── Write tests

Code Focus:
- src/sentinxfl/rag/retriever.py

Deliverables:
✓ Retrieval working
✓ Top-k relevant patterns
```

**Afternoon: RAG Integration**
```
Tasks:
├── Integrate retrieval with LLM
├── Create context injection
├── Test RAG pipeline
├── Compare with non-RAG
└── Bug fixes

Code Focus:
- src/sentinxfl/llm/rag.py

Deliverables:
✓ RAG pipeline complete
✓ Better generations
```

### 5.4 Day 25 (Mar 1): Report Generation

**Morning: Report Templates**
```
Tasks:
├── Create executive report template
├── Create evidence report template
├── Create technical report template
├── Add template rendering
└── Write tests

Code Focus:
- src/sentinxfl/llm/templates.py
- src/sentinxfl/llm/generator.py

Deliverables:
✓ All templates working
✓ Reports generated
```

**Afternoon: Evidence Collection**
```
Tasks:
├── Collect SHAP values for evidence
├── Collect TabNet attention
├── Format evidence package
├── Integrate with generator
└── Write tests

Code Focus:
- src/sentinxfl/llm/evidence.py

Deliverables:
✓ Evidence collected
✓ Injected into prompts
```

### 5.5 Day 26 (Mar 2): Hallucination Guards

**Morning: NLI Verification**
```
Tasks:
├── Load NLI model (small)
├── Implement claim extraction
├── Create entailment checking
├── Add confidence scoring
└── Write tests

Code Focus:
- src/sentinxfl/llm/guards.py

Deliverables:
✓ NLI verification working
✓ Hallucinations detected
```

**Afternoon: Report API**
```
Tasks:
├── Create report API endpoints
├── Add generation endpoint
├── Add history endpoint
├── Integration testing
└── Bug fixes

Code Focus:
- src/sentinxfl/api/routes/reports.py

Deliverables:
✓ Report APIs working
✓ Can generate via API
```

### 5.6 Days 27-28 (Mar 3-4): Testing & Pattern Library

**Weekend Tasks**
```
Tasks:
├── Populate pattern library
├── Full LLM testing
├── Report quality validation
├── Memory profiling
├── Documentation
└── Phase 4 complete

Deliverables:
✓ Pattern library populated
✓ Reports high quality
✓ Phase 4 complete
```

---

## 6. Phase 5: Dashboard (Mar 5-12)

### 6.1 Day 29 (Mar 5): Next.js Setup

**Morning: Project Init**
```
Tasks:
├── Create Next.js 14 project
├── Configure Tailwind CSS
├── Set up shadcn/ui
├── Configure dark theme
├── Create folder structure

Commands:
npx create-next-app@latest frontend
cd frontend
npx shadcn-ui@latest init

Deliverables:
✓ Next.js running
✓ Dark theme applied
```

**Afternoon: Layout Components**
```
Tasks:
├── Create main layout
├── Create sidebar component
├── Create header component
├── Create footer component
├── Set up navigation

Code Focus:
- frontend/app/layout.tsx
- frontend/components/layout/

Deliverables:
✓ Layout complete
✓ Navigation working
```

### 6.2 Day 30 (Mar 6): Executive Overview

**Morning: KPI Components**
```
Tasks:
├── Create KPI card component
├── Create risk score gauge
├── Create ROI metrics cards
├── Style all components
└── Add loading states

Code Focus:
- frontend/components/cards/
- frontend/app/dashboard/page.tsx

Deliverables:
✓ KPI cards working
✓ Risk gauge animated
```

**Afternoon: Charts & Data**
```
Tasks:
├── Create donut chart (risk distribution)
├── Create line chart (timeline)
├── Create changes list
├── Connect to API
└── Add real-time updates

Code Focus:
- frontend/components/charts/
- frontend/lib/api.ts

Deliverables:
✓ Executive Overview complete
✓ Data from API
```

### 6.3 Day 31 (Mar 7): Local Bank View

**Morning: Model Performance**
```
Tasks:
├── Create model comparison table
├── Create performance bar chart
├── Create confusion matrix heatmap
├── Add feature importance chart
└── Style components

Code Focus:
- frontend/app/local-bank/page.tsx

Deliverables:
✓ Model table working
✓ All charts rendering
```

**Afternoon: Training History**
```
Tasks:
├── Create training charts
├── Add loss/metric curves
├── Create ensemble weights display
├── Connect to API
└── Add refresh functionality

Deliverables:
✓ Local Bank View complete
✓ Training history visible
```

### 6.4 Day 32 (Mar 8): Central Knowledge

**Morning: Pattern Display**
```
Tasks:
├── Create executive summary card
├── Create emerging patterns list
├── Create fraud timeline
├── Add severity indicators
└── Style components

Code Focus:
- frontend/app/central-knowledge/page.tsx

Deliverables:
✓ Patterns displayed
✓ Timeline working
```

**Afternoon: Multi-Bank View**
```
Tasks:
├── Create bank selector
├── Add aggregated metrics
├── Create recommended actions
├── Connect to API
└── Bug fixes

Deliverables:
✓ Central Knowledge complete
```

### 6.5 Day 33 (Mar 9): Technical Appendix

**Morning: Privacy Display**
```
Tasks:
├── Create privacy budget tracker
├── Create budget progress bar
├── Create round history table
├── Add DP query history
└── Style components

Code Focus:
- frontend/app/technical/page.tsx

Deliverables:
✓ Privacy budget visible
✓ History tracked
```

**Afternoon: FL Status**
```
Tasks:
├── Create FL status panel
├── Add node status list
├── Create aggregation display
├── Add PII audit display
└── Connect to API

Deliverables:
✓ Technical Appendix complete
```

### 6.6 Day 34 (Mar 10): Export Page

**Morning: Report Generation UI**
```
Tasks:
├── Create report form
├── Add date range picker
├── Add report type selector
├── Add options checkboxes
├── Create generate button

Code Focus:
- frontend/app/export/page.tsx

Deliverables:
✓ Report form working
```

**Afternoon: Report History**
```
Tasks:
├── Create report history table
├── Add download buttons
├── Add preview modal
├── Connect to API
└── Bug fixes

Deliverables:
✓ Export page complete
```

### 6.7 Days 35-36 (Mar 11-12): Polish & Testing

**Weekend Tasks**
```
Tasks:
├── Responsive design fixes
├── Loading states
├── Error handling
├── Cross-browser testing
├── Performance optimization
├── Bug fixes
└── Phase 5 complete

Deliverables:
✓ Dashboard polished
✓ All pages working
✓ Phase 5 complete
```

---

## 7. Phase 6: Integration (Mar 12-19)

### 7.1 End-to-End Integration

```
Tasks:
├── Full pipeline testing
├── Data → PII → Models → FL → Reports → Dashboard
├── Error handling
├── Edge cases
├── Performance optimization
├── Memory management
└── Documentation

Deliverables:
✓ End-to-end working
✓ Demo script ready
```

### 7.2 Demo Script

```python
# scripts/demo.py
"""
SentinXFL Demo Script
Runs complete pipeline for panel demonstration
"""

# 1. Load dataset
# 2. Run PII detection & certification
# 3. Train all models
# 4. Run FL simulation
# 5. Generate reports
# 6. Display dashboard
```

---

## 8. Buffer Period (Mar 19 - May 1)

### 8.1 Publication Prep (Mar 19 - Apr 15)

```
Tasks:
├── Write paper draft
├── Create figures
├── Run experiments
├── Collect results
├── Write related work
├── Proofread & edit
└── Submit to venue

Target Venues:
- IEEE S&P (if strong results)
- PETS (privacy focus)
- AAAI (AI focus)
- Workshop papers as backup
```

### 8.2 Panel Prep (Apr 15 - May 1)

```
Tasks:
├── Prepare presentation slides
├── Practice demo
├── Prepare Q&A answers
├── Backup plans
├── Final polish
└── Panel presentation

Deliverables:
✓ 20-30 minute presentation
✓ Live demo working
✓ Q&A prepared
```

---

## 9. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| VRAM overflow | Medium | High | Strict memory management, batch processing |
| Model accuracy low | Low | High | Ensemble approach, hyperparameter tuning |
| LLM hallucinations | Medium | Medium | Hallucination guards, template constraints |
| Timeline slip | Medium | Medium | Buffer period, MVP focus |
| FL convergence issues | Low | Medium | FedAvg baseline, monitoring |

---

## 10. Success Criteria

### MVP (Mar 19)
- [ ] All 3 datasets loaded and processed
- [ ] PII blocking pipeline working
- [ ] All 4 models + ensemble trained
- [ ] FL simulation with 3+ clients
- [ ] DP budget tracking
- [ ] Report generation working
- [ ] Dashboard complete
- [ ] Demo script running

### Panel Ready (May 1)
- [ ] Polished presentation
- [ ] Live demo prepared
- [ ] Paper draft complete
- [ ] All documentation
- [ ] Q&A prepared

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
