# SentinXFL - Product Requirements Document (PRD)

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)  
> **Supervisor**: Dr. Kiruthika, SRMIST Chennai  

---

## 1. Executive Summary

### 1.1 Product Vision
SentinXFL is a **privacy-first federated fraud detection platform** that enables multiple financial institutions to collaboratively train fraud detection models without sharing raw customer data. The system provides mathematical privacy guarantees through differential privacy, Byzantine-robust aggregation, and a novel **Certified Data Sanitization Pipeline** that blocks PII before any model access.

### 1.2 One-Line Description
> A federated learning platform for multi-institutional fraud detection with certified PII blocking, differential privacy guarantees, and explainable AI reporting.

### 1.3 Core Value Proposition
| Stakeholder | Value |
|-------------|-------|
| **Banks** | Collaborate on fraud detection without sharing customer data |
| **Compliance** | Mathematical proof of privacy (ε,δ)-DP guarantees |
| **Regulators** | Full audit trail, GDPR/DPDP Act compliant |
| **Fraud Analysts** | Real-time alerts with explainable AI reports |

---

## 2. Problem Statement

### 2.1 The Challenge
Financial institutions face a **$30B+ annual fraud problem** but cannot share customer data due to:
- **Privacy regulations**: GDPR, CCPA, India DPDP Act 2023, PCI-DSS
- **Competitive concerns**: Banks don't want to share patterns
- **Legal liability**: Data breaches = massive fines

### 2.2 Current Solutions Fall Short
| Approach | Problem |
|----------|---------|
| Isolated models | Each bank trains alone → weak fraud detection |
| Data sharing agreements | Legal complexity, breach risk, years to negotiate |
| Anonymized data pools | Re-identification attacks possible |
| Third-party data vendors | Trust issues, privacy not guaranteed |

### 2.3 Our Solution
**Federated Learning + Differential Privacy + Certified PII Blocking**
- Raw data NEVER leaves the bank's servers
- Only model gradients shared (with noise)
- Mathematical proof no individual can be identified
- Multi-fraud-type detection (Account, Card, Payment)

---

## 3. Target Users

### 3.1 Primary Users

#### User 1: Fraud Analyst
- **Role**: Monitor fraud alerts, investigate suspicious transactions
- **Needs**: Real-time dashboard, explainable alerts, case management
- **Pain Points**: Alert fatigue, false positives, lack of cross-bank patterns

#### User 2: Data Scientist (Bank)
- **Role**: Train and tune local fraud models
- **Needs**: Model performance metrics, feature importance, drift detection
- **Pain Points**: Limited data, can't learn from other banks' patterns

#### User 3: Compliance Officer
- **Role**: Ensure regulatory compliance, approve data exports
- **Needs**: Privacy budget tracking, audit trails, compliance reports
- **Pain Points**: Proving privacy to regulators, documentation burden

### 3.2 Secondary Users

#### User 4: Executive/CISO
- **Role**: Strategic decisions, budget allocation
- **Needs**: High-level dashboards, ROI metrics, trend reports
- **Pain Points**: Quantifying fraud prevention value

#### User 5: IT Administrator
- **Role**: Deploy and maintain the system
- **Needs**: Health monitoring, easy deployment, security configs
- **Pain Points**: Complex distributed systems, compliance requirements

---

## 4. Functional Requirements

### 4.1 Data Pipeline (FR-DATA)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-DATA-01 | Load CSV files up to 10M rows efficiently | P0 | Use DuckDB + memory mapping |
| FR-DATA-02 | Support 3 dataset schemas (Bank Account, Credit Card, PaySim) | P0 | Unified internal schema |
| FR-DATA-03 | Temporal train/val/test splits (no data leakage) | P0 | Time-based splitting |
| FR-DATA-04 | Schema validation against canonical schema | P0 | Reject invalid data |
| FR-DATA-05 | Incremental data loading for streaming | P1 | Future enhancement |

### 4.2 PII Blocking Pipeline (FR-PII) - **PATENT CORE**

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-PII-01 | Statistical PII detection (entropy, cardinality) | P0 | Auto-detect sensitive columns |
| FR-PII-02 | Pattern matching (CC#, SSN, email, phone) | P0 | Regex-based detection |
| FR-PII-03 | Quasi-identifier combination analysis | P0 | k-anonymity check |
| FR-PII-04 | Hard blocking gate (reject uncertified data) | P0 | **No bypass possible** |
| FR-PII-05 | Safe transformations (binning, generalization) | P0 | Preserve utility |
| FR-PII-06 | Re-identification risk scoring | P0 | Quantify privacy |
| FR-PII-07 | Cryptographic certification with audit log | P0 | Tamper-proof |
| FR-PII-08 | Location hierarchy generalization | P1 | City → State → Country |

### 4.3 Machine Learning Models (FR-ML)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-ML-01 | XGBoost gradient boosting classifier | P0 | CPU only |
| FR-ML-02 | LightGBM gradient boosting classifier | P0 | CPU only |
| FR-ML-03 | Isolation Forest anomaly detection | P0 | CPU only |
| FR-ML-04 | TabNet attention-based neural network | P0 | GPU (~1GB VRAM) |
| FR-ML-05 | Weighted ensemble with learned weights | P0 | Combine all models |
| FR-ML-06 | Model metrics: AUC-ROC, Precision, Recall, F1 | P0 | Standard metrics |
| FR-ML-07 | Inference latency < 100ms (batch), < 10ms (single) | P0 | Performance SLA |
| FR-ML-08 | Model serialization/deserialization | P0 | Save/load models |

### 4.4 Federated Learning (FR-FL)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-FL-01 | Flower-based FL server (coordinator) | P0 | Production framework |
| FR-FL-02 | Flower-based FL client (local node) | P0 | Bank-side component |
| FR-FL-03 | FedAvg aggregation algorithm | P0 | Baseline |
| FR-FL-04 | Byzantine-robust: Multi-Krum | P0 | Tolerate f malicious |
| FR-FL-05 | Byzantine-robust: Trimmed Mean | P0 | Remove outliers |
| FR-FL-06 | Byzantine-robust: Coordinate Median | P0 | Robust to attacks |
| FR-FL-07 | Single-machine FL simulation | P0 | For development/demo |
| FR-FL-08 | Secure aggregation (Shamir's secret sharing) | P1 | Enhanced privacy |
| FR-FL-09 | Contribution fairness (Shapley values) | P2 | Future enhancement |

### 4.5 Differential Privacy (FR-DP)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-DP-01 | RDP (Rényi DP) accountant | P0 | Tight composition |
| FR-DP-02 | Gaussian mechanism for gradients | P0 | Primary mechanism |
| FR-DP-03 | Laplace mechanism for counts/sums | P0 | For statistics |
| FR-DP-04 | Privacy budget tracking (per round) | P0 | ε accumulation |
| FR-DP-05 | Budget visualization in dashboard | P0 | Real-time display |
| FR-DP-06 | Automatic query rejection when exhausted | P0 | Hard budget limit |
| FR-DP-07 | (ε,δ)-DP conversion from RDP | P0 | Standard format |
| FR-DP-08 | Gradient clipping (L2 norm) | P0 | Sensitivity bound |

### 4.6 Explainable AI & LLM (FR-XAI)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-XAI-01 | Local LLM: Phi-3-mini-4k (4-bit quantized) | P0 | ~2GB VRAM |
| FR-XAI-02 | ChromaDB vector store for patterns | P0 | RAG retrieval |
| FR-XAI-03 | SHAP values for feature importance | P0 | Model explanation |
| FR-XAI-04 | Report generation from templates | P0 | Structured output |
| FR-XAI-05 | Hallucination guards (NLI verification) | P0 | Factual accuracy |
| FR-XAI-06 | Executive report (sanitized, high-level) | P0 | For stakeholders |
| FR-XAI-07 | Evidence report (detailed, auditable) | P0 | For compliance |
| FR-XAI-08 | Confidence scoring on explanations | P1 | Uncertainty |

### 4.7 Dashboard (FR-UI)

| ID | Requirement | Priority | Notes |
|----|-------------|----------|-------|
| FR-UI-01 | Dark theme matching reference design | P0 | Professional look |
| FR-UI-02 | Executive Overview page | P0 | KPIs, trends |
| FR-UI-03 | Local Bank View page | P0 | Bank-specific metrics |
| FR-UI-04 | Central Knowledge page | P0 | Aggregated patterns |
| FR-UI-05 | Technical Appendix page | P0 | Model details |
| FR-UI-06 | Export page | P0 | Report downloads |
| FR-UI-07 | Real-time metrics updates | P1 | WebSocket/polling |
| FR-UI-08 | Mobile responsive design | P1 | Tablet support |

---

## 5. Non-Functional Requirements

### 5.1 Performance (NFR-PERF)

| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-PERF-01 | Load 1M rows | < 30 seconds | DuckDB + lazy loading |
| NFR-PERF-02 | Train ensemble (1M rows) | < 5 minutes | With 4GB VRAM constraint |
| NFR-PERF-03 | Single inference | < 10ms | Real-time scoring |
| NFR-PERF-04 | Batch inference (1000 rows) | < 100ms | Bulk processing |
| NFR-PERF-05 | FL round completion | < 60 seconds | Simulation mode |
| NFR-PERF-06 | Dashboard initial load | < 3 seconds | First contentful paint |
| NFR-PERF-07 | LLM report generation | < 30 seconds | Phi-3 4-bit |

### 5.2 Resource Constraints (NFR-RES)

| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-RES-01 | GPU VRAM usage | < 4GB | RTX 3050 constraint |
| NFR-RES-02 | Peak RAM usage | < 12GB | Leave headroom for OS |
| NFR-RES-03 | Disk space (models) | < 5GB | All models + LLM |
| NFR-RES-04 | CPU cores utilized | ≤ 6 | Efficient parallelism |

### 5.3 Security (NFR-SEC)

| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-SEC-01 | No PII in model inputs | 100% blocked | Certified pipeline |
| NFR-SEC-02 | Audit log integrity | Tamper-proof | Hash chain |
| NFR-SEC-03 | API authentication | JWT tokens | Basic auth for MVP |
| NFR-SEC-04 | HTTPS only | Enforced | TLS 1.3 |

### 5.4 Privacy (NFR-PRIV)

| ID | Requirement | Target | Notes |
|----|-------------|--------|-------|
| NFR-PRIV-01 | Privacy budget (ε) | < 1.0 | Production threshold |
| NFR-PRIV-02 | Delta (δ) | < 1e-5 | Negligible |
| NFR-PRIV-03 | k-anonymity | k ≥ 5 | Re-identification protection |
| NFR-PRIV-04 | Re-identification risk | < 5% | Measured |

### 5.5 Compliance (NFR-COMP)

| ID | Requirement | Notes |
|----|-------------|-------|
| NFR-COMP-01 | GDPR (EU) | Right to erasure, consent |
| NFR-COMP-02 | CCPA (California) | Data disclosure, opt-out |
| NFR-COMP-03 | India DPDP Act 2023 | Data localization, consent |
| NFR-COMP-04 | PCI-DSS | Card data protection |
| NFR-COMP-05 | RBI Guidelines | Indian banking compliance |
| NFR-COMP-06 | GLBA | US financial privacy |

---

## 6. Dataset Specifications

### 6.1 Bank Account Fraud Dataset
- **Source**: Kaggle (synthetic)
- **Rows**: 6,000,000 (6 variants × 1M each)
- **Columns**: 32
- **Target**: `fraud_bool`
- **Fraud Type**: Account opening fraud

### 6.2 Credit Card Fraud Dataset
- **Source**: Kaggle (ULB - real anonymized)
- **Rows**: 284,807
- **Columns**: 31 (V1-V28 PCA + Time + Amount + Class)
- **Target**: `Class`
- **Fraud Type**: Card transaction fraud

### 6.3 PaySim Dataset
- **Source**: Kaggle (synthetic mobile payments)
- **Rows**: 6,362,620
- **Columns**: 11
- **Target**: `isFraud`
- **Fraud Type**: Mobile payment fraud (Transfer, Cash-out)

### 6.4 Total Coverage
- **Total Transactions**: 12.6M+
- **Fraud Types**: 3 (Account Opening, Card Transaction, Mobile Payment)

---

## 7. Success Metrics

### 7.1 Model Performance
| Metric | Target | Notes |
|--------|--------|-------|
| AUC-ROC | > 0.95 | Ensemble model |
| Precision | > 0.90 | At 80% recall |
| Recall | > 0.80 | Catch most fraud |
| F1-Score | > 0.85 | Balanced metric |

### 7.2 Privacy Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| Final ε | < 1.0 | After all FL rounds |
| Re-identification risk | < 5% | k-anonymity verified |
| PII leakage | 0 | Blocked by design |

### 7.3 System Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| Uptime | > 99% | For demo/panel |
| Test coverage | > 80% | Unit + integration |
| Documentation | Complete | All modules documented |

---

## 8. Out of Scope (MVP)

The following are **NOT** included in the MVP:
- Real multi-machine FL deployment (simulation only)
- Mobile app
- Real-time streaming data ingestion
- Advanced secure aggregation (homomorphic encryption)
- Production-grade authentication (OAuth2, SSO)
- Multi-language support
- Automated model retraining pipeline

---

## 9. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| VRAM overflow | High | Medium | Strict memory profiling, batch processing |
| Model accuracy too low | High | Low | Ensemble approach, hyperparameter tuning |
| LLM hallucinations | Medium | Medium | NLI verification, template constraints |
| FL convergence issues | Medium | Medium | FedAvg baseline, monitoring |
| Timeline slip | Medium | Medium | Buffer period, MVP scope control |

---

## 10. Timeline Overview

| Phase | Duration | Dates | Key Deliverables |
|-------|----------|-------|------------------|
| Phase 1: Foundation | 1 week | Feb 5-12 | Data pipeline, PII blocking |
| Phase 2: ML Models | 1 week | Feb 12-19 | All 4 models + ensemble |
| Phase 3: FL + DP | 1 week | Feb 19-26 | Flower FL, DP accountant |
| Phase 4: LLM | 1 week | Feb 26-Mar 5 | Phi-3, RAG, reports |
| Phase 5: Dashboard | 1 week | Mar 5-12 | All 5 pages |
| Phase 6: Integration | 1 week | Mar 12-19 | End-to-end working |
| Buffer | 6 weeks | Mar 19-May 1 | Polish, publication, panel prep |

**Hard Deadline**: May 1, 2026 (Panel Presentation)

---

## 11. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Project Lead | Anshuman Bakshi | Feb 5, 2026 | ____________ |
| Contributor | Komal | Feb 5, 2026 | ____________ |
| Supervisor | Dr. Kiruthika | ____________ | ____________ |

---

*Document Version History*
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Feb 5, 2026 | Anshuman Bakshi | Initial PRD |
| 2.0 | Feb 5, 2026 | Anshuman Bakshi | Optimized for laptop constraints |
