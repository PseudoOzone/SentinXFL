# SentinXFL - Regulatory Compliance

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)

---

## 1. Compliance Overview

SentinXFL is designed to comply with the following regulations:

| Regulation | Jurisdiction | Relevance | Priority |
|------------|--------------|-----------|----------|
| GDPR | EU | Personal data protection | P0 |
| DPDP Act 2023 | India | Personal data protection | P0 |
| RBI Guidelines | India | Banking data security | P0 |
| PCI-DSS | Global | Payment card data | P1 |
| CCPA | California, US | Consumer privacy | P1 |
| SOC 2 Type II | Global | Security controls | P2 |

---

## 2. GDPR Compliance

### 2.1 Applicable Articles

| Article | Requirement | SentinXFL Implementation |
|---------|-------------|-------------------------|
| Art. 5(1)(a) | Lawfulness, fairness, transparency | Clear purpose documentation, consent tracking |
| Art. 5(1)(b) | Purpose limitation | Data used only for fraud detection |
| Art. 5(1)(c) | Data minimization | PII blocking removes unnecessary data |
| Art. 5(1)(e) | Storage limitation | Configurable retention periods |
| Art. 5(1)(f) | Integrity & confidentiality | Encryption, access controls, audit logs |
| Art. 25 | Data protection by design | Privacy-first architecture |
| Art. 32 | Security of processing | DP guarantees, encrypted communication |
| Art. 35 | DPIA required | High-risk processing → DPIA template provided |

### 2.2 Data Subject Rights

```yaml
gdpr_rights:
  right_to_access:
    implementation: "Export API with PII masking"
    api_endpoint: "GET /api/v1/compliance/subject/{id}/data"
    response_time: "30 days max"
    
  right_to_rectification:
    implementation: "Not applicable - no PII stored after blocking"
    status: "N/A"
    
  right_to_erasure:
    implementation: "Certificate revocation + audit purge"
    api_endpoint: "DELETE /api/v1/compliance/subject/{id}"
    response_time: "30 days max"
    
  right_to_portability:
    implementation: "JSON/CSV export of non-PII data"
    api_endpoint: "GET /api/v1/compliance/subject/{id}/export"
    format: "JSON, CSV"
    
  right_to_object:
    implementation: "Opt-out flag in processing pipeline"
    api_endpoint: "POST /api/v1/compliance/subject/{id}/opt-out"
```

### 2.3 Lawful Basis

```
Lawful Basis for Processing: LEGITIMATE INTEREST (Art. 6(1)(f))

Justification:
├── Fraud detection protects data subjects from financial harm
├── Banks have legitimate interest in preventing losses
├── Processing is necessary - cannot detect fraud without transaction analysis
├── Minimal impact on data subjects - PII is blocked before model training
└── Balancing test: Fraud prevention benefit > Privacy impact (with DP guarantees)

Documentation Required:
├── Legitimate Interest Assessment (LIA)
├── Data Protection Impact Assessment (DPIA)
└── Records of Processing Activities (ROPA)
```

---

## 3. India DPDP Act 2023 Compliance

### 3.1 Key Requirements

| Section | Requirement | SentinXFL Implementation |
|---------|-------------|-------------------------|
| Sec. 4 | Consent for processing | Consent framework with purpose binding |
| Sec. 5 | Purpose limitation | Fraud detection purpose locked |
| Sec. 6 | Data minimization | 5-gate PII blocking |
| Sec. 8 | Data accuracy | Validation rules on input |
| Sec. 9 | Storage limitation | Configurable retention |
| Sec. 11 | Data Principal rights | Export/deletion APIs |
| Sec. 18 | Significant Data Fiduciary | Enhanced compliance for banks |

### 3.2 Consent Management

```python
# Consent tracking schema
consent_record = {
    "consent_id": "uuid",
    "data_principal_id": "hashed_identifier",
    "purpose": "fraud_detection",
    "consent_given": True,
    "consent_timestamp": "2026-02-05T10:00:00Z",
    "consent_mechanism": "explicit_checkbox",
    "withdrawable": True,
    "withdrawal_timestamp": None,
    "data_fiduciary": "bank_name",
    "processing_purposes": [
        "fraud_pattern_detection",
        "model_training",
        "risk_scoring"
    ],
    "retention_period_days": 365
}
```

### 3.3 Data Principal Rights (Indian Context)

```yaml
dpdp_rights:
  right_to_information:
    requirement: "Summary of processing activities"
    implementation: "Privacy notice endpoint"
    api: "GET /api/v1/compliance/privacy-notice"
    
  right_to_correction_erasure:
    requirement: "Correct or erase personal data"
    implementation: "Certificate revocation"
    api: "POST /api/v1/compliance/subject/{id}/correct"
    
  right_to_grievance_redressal:
    requirement: "Complaint mechanism"
    implementation: "Grievance tracking system"
    api: "POST /api/v1/compliance/grievance"
    
  right_to_nominate:
    requirement: "Nominate person in case of death/incapacity"
    implementation: "Nominee registration"
    api: "POST /api/v1/compliance/subject/{id}/nominee"
```

---

## 4. RBI Guidelines Compliance

### 4.1 Master Direction on IT Governance

| Clause | Requirement | Implementation |
|--------|-------------|----------------|
| 5.1 | IT Governance Framework | Documented architecture |
| 6.1 | Information Security Policy | Security.md + controls |
| 7.1 | IT Operations | Logging, monitoring, alerts |
| 8.1 | IS Audit | Audit trail with hash chain |
| 9.1 | Business Continuity | Backup, recovery procedures |
| 10.1 | IT Outsourcing | N/A (self-hosted) |

### 4.2 Data Localization

```yaml
rbi_data_localization:
  requirement: "Payment data must be stored in India"
  
  compliance_approach:
    storage:
      - "All DuckDB files stored on Indian servers"
      - "ChromaDB vectors stored locally"
      - "No cloud storage for raw data"
    
    processing:
      - "Model training on Indian infrastructure"
      - "FL aggregation can be cross-border (only gradients)"
      - "Gradients contain no personal data (DP noise)"
    
    transfer:
      - "Raw data: NEVER transferred"
      - "Gradients: Transferred with DP noise (compliant)"
      - "Reports: Generated locally"
```

### 4.3 Fraud Monitoring Requirements

```yaml
rbi_fraud_monitoring:
  circular: "RBI/2016-17/217 dated 29th September 2016"
  
  requirements:
    real_time_monitoring:
      implementation: "Streaming inference API"
      latency: "< 100ms"
      
    fraud_reporting:
      implementation: "Compliance report generator"
      format: "As per RBI format"
      frequency: "Monthly + on-demand"
      
    suspicious_transaction_reporting:
      implementation: "Automated STR generation"
      threshold: "Configurable per bank"
      
    customer_alert:
      implementation: "Alert API for downstream systems"
      channels: "API webhook, SMS integration ready"
```

---

## 5. PCI-DSS Compliance

### 5.1 Relevant Requirements

| Requirement | Description | Implementation |
|-------------|-------------|----------------|
| 3.4 | Render PAN unreadable | Credit card numbers never stored |
| 4.1 | Encrypt transmission | TLS 1.3 for all API calls |
| 6.5 | Secure coding | OWASP top 10 addressed |
| 7.1 | Access control | Role-based access |
| 8.2 | Authentication | JWT tokens with expiry |
| 10.1 | Audit trails | Hash-chained audit log |
| 12.1 | Security policy | Documented in Security.md |

### 5.2 Cardholder Data Handling

```
PCI-DSS Scope Reduction Strategy:

SentinXFL NEVER stores:
├── Primary Account Number (PAN)
├── Cardholder Name
├── Expiration Date
├── Service Code
├── CVV/CVC
└── PIN

Credit Card Dataset (Kaggle):
├── Already PCA-transformed
├── No cardholder data present
├── V1-V28 are not reversible
└── Amount is the only raw field

Result: SentinXFL is OUT OF PCI-DSS SCOPE for cardholder data
```

---

## 6. Privacy Engineering

### 6.1 Differential Privacy Guarantees

```
Mathematical Privacy Guarantee:

For any two adjacent datasets D and D' differing in one record:
    Pr[M(D) ∈ S] ≤ e^ε × Pr[M(D') ∈ S] + δ

SentinXFL Parameters:
├── ε (epsilon) = 1.0 (strong privacy)
├── δ (delta) = 1e-5 (negligible failure probability)
└── Composition: RDP accounting for tight bounds

Interpretation:
├── An adversary learns almost nothing about any individual
├── Even with auxiliary information, re-identification is infeasible
└── Mathematical proof, not just policy
```

### 6.2 k-Anonymity Implementation

```python
# k-Anonymity verification
def verify_k_anonymity(df: pd.DataFrame, quasi_ids: list, k: int = 5) -> bool:
    """
    Verify that every combination of quasi-identifiers
    appears at least k times in the dataset.
    """
    equivalence_classes = df.groupby(quasi_ids).size()
    return equivalence_classes.min() >= k

# SentinXFL configuration
k_anonymity_config = {
    "minimum_k": 5,
    "quasi_identifiers": [
        "income_bin",      # Binned income
        "age_bin",         # Binned age
        "employment_cat",  # Generalized employment
        "housing_cat"      # Generalized housing
    ],
    "action_on_violation": "block",  # Prevent data release
    "warning_threshold_k": 10        # Warn if k < 10
}
```

### 6.3 PII Blocking Pipeline (Patent Core)

```
5-Gate Certified Data Sanitization:

Gate 1: Statistical Detection
├── Entropy analysis (H > 4.0 → PII candidate)
├── Cardinality ratio (unique/total > 0.9 → PII)
└── Output: PII column candidates

Gate 2: Pattern Matching
├── Regex patterns (SSN, CC, email, phone, Aadhaar, PAN)
├── Format detection (dates, codes, IDs)
└── Output: Pattern-matched PII columns

Gate 3: Quasi-Identifier Analysis
├── k-Anonymity calculation
├── l-Diversity check
├── Re-identification risk scoring
└── Output: QID combinations to transform

Gate 4: Transformation
├── Binning (continuous → categorical)
├── Generalization (specific → general)
├── Suppression (remove high-risk)
├── DP noise (Laplace/Gaussian)
└── Output: Transformed safe data

Gate 5: Hard Blocking
├── Final regex scan
├── Embedding similarity check
├── Certificate generation
└── Output: Certified safe data + certificate

INVARIANT: No PII passes Gate 5. Ever.
```

---

## 7. Compliance Monitoring

### 7.1 Audit Trail Schema

```python
audit_entry = {
    "entry_id": "uuid",
    "timestamp": "ISO 8601",
    "action": "pii_detection | transformation | certification | access | export",
    "actor": "system | user_id",
    "resource": "dataset_id | certificate_id | model_id",
    "details": {
        "columns_processed": ["col1", "col2"],
        "pii_detected": ["income", "age"],
        "transformations": [
            {"column": "income", "type": "binning", "bins": 4},
            {"column": "age", "type": "binning", "bins": 5}
        ],
        "result": "certified | blocked"
    },
    "previous_hash": "sha256_of_previous_entry",
    "entry_hash": "sha256_of_this_entry"
}
```

### 7.2 Compliance Dashboard Metrics

```yaml
compliance_metrics:
  privacy_budget:
    description: "Remaining DP epsilon"
    current: 0.7
    total: 1.0
    status: "healthy"
    
  k_anonymity:
    description: "Minimum equivalence class size"
    current: 8
    minimum: 5
    status: "healthy"
    
  pii_blocks:
    description: "PII columns blocked this month"
    count: 47
    categories: {direct: 12, quasi: 35}
    
  consent_coverage:
    description: "Data with valid consent"
    percentage: 99.2
    expired: 0.8
    
  audit_trail:
    description: "Audit entries (30 days)"
    count: 12847
    integrity: "verified"
    last_check: "2026-02-05T09:00:00Z"
```

---

## 8. Compliance Reports

### 8.1 Available Reports

| Report | Frequency | Format | API Endpoint |
|--------|-----------|--------|--------------|
| Privacy Budget Status | Real-time | JSON | GET /api/v1/compliance/dp-budget |
| PII Blocking Summary | Daily | PDF, CSV | GET /api/v1/compliance/pii-report |
| Audit Trail Export | On-demand | JSON, CSV | GET /api/v1/compliance/audit-export |
| DPIA Report | Quarterly | PDF | GET /api/v1/compliance/dpia |
| RBI Fraud Report | Monthly | As per RBI format | GET /api/v1/compliance/rbi-report |
| Consent Status | On-demand | JSON | GET /api/v1/compliance/consent-status |

### 8.2 Automated Compliance Alerts

```yaml
compliance_alerts:
  privacy_budget_low:
    condition: "epsilon_remaining < 0.2"
    action: "email_compliance_officer"
    severity: "warning"
    
  privacy_budget_exhausted:
    condition: "epsilon_remaining <= 0"
    action: "block_queries + alert"
    severity: "critical"
    
  k_anonymity_violation:
    condition: "k < minimum_k"
    action: "block_data_release + alert"
    severity: "critical"
    
  audit_integrity_failure:
    condition: "hash_chain_broken"
    action: "freeze_system + alert_security"
    severity: "critical"
    
  consent_expiring:
    condition: "consent_expires_in < 30_days"
    action: "notify_for_renewal"
    severity: "info"
```

---

## 9. Compliance Checklist

### 9.1 Pre-Deployment Checklist

- [ ] Privacy Impact Assessment (DPIA) completed
- [ ] Legitimate Interest Assessment (LIA) documented
- [ ] Records of Processing Activities (ROPA) created
- [ ] Data Processing Agreement (DPA) with any third parties
- [ ] Privacy Notice updated and accessible
- [ ] Consent mechanism implemented and tested
- [ ] PII blocking pipeline validated
- [ ] DP parameters configured (ε=1.0, δ=1e-5)
- [ ] Audit trail enabled and verified
- [ ] Access controls configured
- [ ] Encryption enabled (TLS 1.3)
- [ ] Incident response plan documented
- [ ] Data retention policy configured
- [ ] Compliance dashboard accessible

### 9.2 Periodic Review Checklist (Monthly)

- [ ] Review privacy budget consumption
- [ ] Verify k-anonymity levels
- [ ] Audit trail integrity check
- [ ] Consent status review
- [ ] Access log review
- [ ] Security vulnerability scan
- [ ] Compliance report generation
- [ ] Policy update review

---

## 10. Incident Response

### 10.1 Data Breach Protocol

```yaml
breach_response:
  detection:
    - "Automated monitoring alerts"
    - "Audit log anomaly detection"
    - "User reports"
    
  containment:
    - "Isolate affected systems"
    - "Revoke compromised certificates"
    - "Block suspicious access"
    
  notification:
    - "72 hours to DPA (GDPR Art. 33)"
    - "72 hours to DPB (DPDP Sec. 12)"
    - "Affected data subjects without undue delay"
    
  documentation:
    - "Incident timeline"
    - "Affected data categories"
    - "Mitigation measures"
    - "Prevention improvements"
```

### 10.2 Privacy Impact Mitigation

```
If PII breach suspected:

1. IMMEDIATE (0-1 hour):
   └── Check: Did any PII pass the 5-gate pipeline?
       ├── If NO: Verify via audit trail hash chain
       └── If YES: This indicates pipeline failure (critical bug)

2. INVESTIGATION (1-24 hours):
   ├── Which gate failed?
   ├── What data was exposed?
   ├── Who had access?
   └── Timeline of events

3. REMEDIATION:
   ├── Patch pipeline vulnerability
   ├── Revoke affected certificates
   ├── Re-process affected data
   └── Update detection rules

4. NOTIFICATION:
   ├── Regulators (GDPR: 72h, DPDP: 72h)
   ├── Affected individuals (if high risk)
   └── Internal stakeholders

Note: Due to DP guarantees, even if model gradients were exposed,
      individual data cannot be reconstructed (mathematical guarantee).
```

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
