# SentinXFL - Project Glossary

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)

---

## Quick Reference

### Acronyms

| Acronym | Full Form | Description |
|---------|-----------|-------------|
| **AUC** | Area Under Curve | Model performance metric (0-1) |
| **CCPA** | California Consumer Privacy Act | US privacy regulation |
| **DP** | Differential Privacy | Mathematical privacy guarantee |
| **DPDPA** | Digital Personal Data Protection Act | India's 2023 privacy law |
| **FL** | Federated Learning | Distributed ML without data sharing |
| **GDPR** | General Data Protection Regulation | EU privacy regulation |
| **LLM** | Large Language Model | AI for text generation |
| **MFA** | Multi-Factor Authentication | Two-step verification |
| **NLI** | Natural Language Inference | Text entailment checking |
| **PII** | Personally Identifiable Information | Data identifying individuals |
| **QID** | Quasi-Identifier | Indirect identifying data |
| **RAG** | Retrieval Augmented Generation | LLM + knowledge retrieval |
| **RBAC** | Role-Based Access Control | Permission management |
| **RBI** | Reserve Bank of India | Indian banking regulator |
| **RDP** | Rényi Differential Privacy | Tighter DP composition |
| **SHAP** | SHapley Additive exPlanations | Model explanation method |
| **SMOTE** | Synthetic Minority Over-sampling Technique | Class balancing |
| **VRAM** | Video Random Access Memory | GPU memory |
| **XAI** | Explainable AI | Interpretable machine learning |

---

## Technical Terms

### A

**Aggregation (FL)**
: Combining model updates from multiple clients. SentinXFL supports FedAvg, Multi-Krum, Trimmed Mean, and Coordinate Median.

**Attention Weights (TabNet)**
: Importance scores for features at each decision step. Used for explainability.

**Audit Trail**
: Hash-chained log of all system operations for compliance verification.

### B

**Batch Size**
: Number of samples processed together. SentinXFL uses 1024 for TabNet to fit in 4GB VRAM.

**Byzantine Fault Tolerance**
: Ability to operate correctly despite malicious participants. Multi-Krum tolerates f Byzantine clients.

**Binning**
: Converting continuous values to categorical ranges (e.g., age → [18-25, 26-35, ...]).

### C

**Cardinality**
: Number of unique values in a column. High cardinality (>90% unique) suggests PII.

**Certificate (PII)**
: Cryptographic proof that data passed all 5 gates of the sanitization pipeline.

**Class Imbalance**
: When fraud cases are rare (0.1-1.4% in our datasets). Addressed with class weights.

**Clipping (Gradient)**
: Bounding gradient norms to limit sensitivity for differential privacy.

**Composability (DP)**
: Property that allows tracking total privacy cost across multiple queries.

### D

**Data Fiduciary**
: Entity determining purpose of data processing (DPDPA term for data controller).

**Data Principal**
: Individual whose data is processed (DPDPA term for data subject).

**Delta (δ)**
: Failure probability in differential privacy. SentinXFL uses δ = 10⁻⁵.

**DuckDB**
: Embedded analytical database. Used for efficient large dataset processing.

### E

**Early Stopping**
: Stopping training when validation metric stops improving. Prevents overfitting.

**Embedding**
: Dense vector representation of text. Used in RAG for semantic search.

**Ensemble**
: Combining multiple models. SentinXFL uses weighted average of XGBoost, LightGBM, TabNet, IsolationForest.

**Entropy**
: Measure of randomness. High entropy (>4.0) indicates potential PII.

**Epsilon (ε)**
: Privacy loss parameter in DP. Lower = more private. SentinXFL uses ε = 1.0.

### F

**Feature Engineering**
: Creating new features from raw data (e.g., velocity ratios, risk scores).

**FedAvg**
: Federated Averaging - basic FL aggregation that averages client gradients.

**Flower**
: FL framework used by SentinXFL. Handles client-server communication.

### G

**Gaussian Mechanism**
: Adding Gaussian noise for DP. Noise σ = sensitivity × √(2 ln(1.25/δ)) / ε.

**Generalization (PII)**
: Making values less specific (e.g., "Software Engineer" → "Employed").

**Gradient**
: Model update computed from training data. Shared in FL instead of raw data.

### H

**Hallucination**
: LLM generating false or unsupported claims. Detected via NLI verification.

**Hash Chain**
: Linking records via cryptographic hashes. Tampering any record breaks the chain.

**Hyperparameter**
: Model configuration (learning rate, depth, etc.) set before training.

### I

**Inference**
: Using trained model to make predictions on new data.

**IsolationForest**
: Anomaly detection algorithm. Isolates anomalies via random partitioning.

### J

**JWT**
: JSON Web Token. Used for API authentication with short expiry (15 min).

### K

**k-Anonymity**
: Privacy property where each record is indistinguishable from k-1 others.

**Krum (Multi-Krum)**
: Byzantine-robust aggregation selecting gradients closest to neighbors.

### L

**Laplace Mechanism**
: Adding Laplace noise for DP. Alternative to Gaussian mechanism.

**Lazy Loading**
: Loading data only when needed. DuckDB uses this for memory efficiency.

**LightGBM**
: Gradient boosting framework. Faster than XGBoost, used on CPU.

**Logpoint**
: LLM-generated message logged instead of breakpoint stopping execution.

### M

**Membership Inference**
: Attack determining if record was in training data. Prevented by DP.

**MLflow**
: Experiment tracking platform. Logs metrics, models, and parameters.

**Model Inversion**
: Attack reconstructing training data from model. Prevented by DP.

**Model Poisoning**
: Malicious client sending bad gradients. Prevented by Byzantine-robust aggregation.

### N

**NLI (Natural Language Inference)**
: Determining if hypothesis is entailed by premise. Used to verify LLM claims.

**Noise (DP)**
: Random values added to query results for privacy. Calibrated to sensitivity.

### O

**Opacus**
: PyTorch library for differentially private training.

**Overfitting**
: Model memorizing training data instead of learning patterns.

### P

**Phi-3**
: Microsoft's small LLM (3.8B parameters). Used with 4-bit quantization (~2GB VRAM).

**Polars**
: Fast DataFrame library. Replacement for pandas with better performance.

**Precision**
: True positives / (True positives + False positives). Important for fraud detection.

**Privacy Budget**
: Total allowable privacy loss. Once exhausted, no more queries allowed.

### Q

**Quantization**
: Reducing model precision (32-bit → 4-bit). Reduces VRAM usage 8x.

**Quasi-Identifier**
: Combination of attributes that could identify someone (age + ZIP + gender).

### R

**RAG (Retrieval Augmented Generation)**
: Enhancing LLM with retrieved context. Reduces hallucination.

**Recall**
: True positives / (True positives + False negatives). Catching all frauds.

**Re-identification**
: Linking anonymized data back to individuals. Prevented by k-anonymity + DP.

**RDP (Rényi DP)**
: Tighter privacy accounting using Rényi divergence. Better composition.

**ROC Curve**
: Receiver Operating Characteristic. Plots TPR vs FPR at various thresholds.

### S

**Sensitivity (DP)**
: Maximum change in query output from one record. Must be bounded for DP.

**SHAP**
: Game-theoretic explanation method assigning importance to each feature.

**Stratified Split**
: Train/test split maintaining class distribution. Essential for imbalanced data.

**Suppression (PII)**
: Removing high-risk columns entirely from output.

### T

**TabNet**
: Neural network for tabular data with attention mechanism. ~1GB VRAM.

**Temporal Split**
: Splitting by time (train on past, test on future). Prevents data leakage.

**Threshold (Classification)**
: Probability cutoff for labeling as fraud. Optimized for precision/recall tradeoff.

**Trimmed Mean**
: Aggregation discarding extreme values. Robust to outlier gradients.

### U

**Undersampling**
: Reducing majority class size for balance. Used for very large datasets.

### V

**Validation Set**
: Data for tuning hyperparameters. Separate from test set.

**Vector Store**
: Database for embeddings with similarity search. ChromaDB in SentinXFL.

### W

**Weighted Ensemble**
: Combining models with different weights. Weights optimized on validation set.

### X

**XGBoost**
: Extreme Gradient Boosting. State-of-the-art for tabular data.

---

## Key Formulas

### Differential Privacy

**Gaussian Mechanism:**
$$\mathcal{M}(x) = f(x) + \mathcal{N}(0, \sigma^2)$$
where $\sigma = \frac{\text{sensitivity} \cdot \sqrt{2\ln(1.25/\delta)}}{\epsilon}$

**DP Guarantee:**
$$\Pr[\mathcal{M}(D) \in S] \leq e^\epsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

### Information Entropy

$$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

### k-Anonymity Risk

$$\text{Risk} = \frac{1}{k_{\min}}$$
where $k_{\min}$ is the smallest equivalence class.

### AUC-ROC

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d\text{FPR}(t)$$

---

## Dataset Quick Reference

| Dataset | Rows | Fraud % | PII Columns | Target |
|---------|------|---------|-------------|--------|
| Bank Account Fraud | 6M | 1.4% | income, age, employment, housing | fraud_bool |
| Credit Card | 285K | 0.17% | None (PCA transformed) | Class |
| PaySim | 6.3M | 0.13% | nameOrig, nameDest | isFraud |

---

## Model Quick Reference

| Model | Type | VRAM | Training Time* | AUC-ROC |
|-------|------|------|----------------|---------|
| XGBoost | Boosting | CPU | ~2 min | 0.92+ |
| LightGBM | Boosting | CPU | ~1 min | 0.92+ |
| IsolationForest | Anomaly | CPU | ~30 sec | 0.85+ |
| TabNet | Neural | 1GB | ~10 min | 0.91+ |
| Ensemble | Combined | CPU | ~1 min | 0.94+ |

*On 100K sample, RTX 3050

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
