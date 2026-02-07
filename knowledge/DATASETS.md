# SentinXFL - Dataset Documentation

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)

---

## 1. Dataset Overview

| Dataset | Rows | Fraud % | Source | License |
|---------|------|---------|--------|---------|
| Bank Account Fraud | 6,000,000 | ~1.4% | Kaggle (NVS Yashwanth) | CC0 |
| Credit Card Fraud | 284,807 | 0.17% | Kaggle (MLG-ULB) | ODbL |
| PaySim | 6,362,620 | 0.13% | Kaggle (Edgar Lopez-Rojas) | CC BY-SA 4.0 |
| **Total** | **12,647,427** | ~0.5% avg | - | - |

---

## 2. Dataset Locations

```yaml
datasets:
  # ALL DATASETS IN PROJECT FOLDER
  base_path: "data/datasets/"  # Relative to project root
  absolute_path: "c:/Users/anshu/SentinXFL_Final/data/datasets/"
  
  bank_account_fraud:
    files:
      - Base.csv              # ~1M rows (203.5 MB)
      - Variant I.csv         # ~1M rows (203.5 MB)
      - Variant II.csv        # ~1M rows (203.6 MB)
      - Variant III.csv       # ~1M rows (240.5 MB)
      - Variant IV.csv        # ~1M rows (203.6 MB)
      - Variant V.csv         # ~1M rows (240.5 MB)
    total_rows: 6,000,000
    total_size: ~1.3 GB
    
  credit_card_fraud:
    files:
      - creditcard.csv        # 284,807 rows (143.8 MB)
    total_rows: 284,807
    total_size: ~144 MB
    
  paysim:
    files:
      - PS_20174392719_1491204439457_log.csv  # 6,362,620 rows (470.7 MB)
    total_rows: 6,362,620
    total_size: ~471 MB
    
  total_size: ~1.9 GB
```

---

## 3. Bank Account Fraud Dataset

### 3.1 Schema

| Column | Type | Description | PII Risk | Action |
|--------|------|-------------|----------|--------|
| `income` | float | Annual income | **HIGH** (QID) | Bin to ranges |
| `name_email_similarity` | float | Name-email matching score | MEDIUM | Keep (0-1 bounded) |
| `prev_address_months_count` | int | Months at previous address | LOW | Keep |
| `current_address_months_count` | int | Months at current address | LOW | Keep |
| `customer_age` | int | Customer age in years | **HIGH** (QID) | Bin to ranges |
| `days_since_request` | float | Days since account request | LOW | Keep |
| `intended_balcon_amount` | float | Intended balance amount | MEDIUM | Keep |
| `payment_type` | string | Payment method type | LOW | Keep |
| `zip_count_4w` | int | ZIP requests in 4 weeks | MEDIUM | Keep |
| `velocity_6h` | float | Transaction velocity 6h | LOW | Keep |
| `velocity_24h` | float | Transaction velocity 24h | LOW | Keep |
| `velocity_4w` | float | Transaction velocity 4w | LOW | Keep |
| `bank_branch_count_8w` | int | Branches used in 8 weeks | LOW | Keep |
| `date_of_birth_distinct_emails_4w` | int | Emails per DOB in 4w | MEDIUM | Keep |
| `employment_status` | string | Employment category | **HIGH** (QID) | Generalize |
| `credit_risk_score` | int | Credit score | MEDIUM | Keep |
| `email_is_free` | bool | Free email domain | LOW | Keep |
| `housing_status` | string | Housing type | **HIGH** (QID) | Generalize |
| `phone_home_valid` | bool | Valid home phone | LOW | Keep |
| `phone_mobile_valid` | bool | Valid mobile phone | LOW | Keep |
| `bank_months_count` | int | Months with bank | LOW | Keep |
| `has_other_cards` | bool | Has other credit cards | LOW | Keep |
| `proposed_credit_limit` | float | Proposed credit limit | MEDIUM | Keep |
| `foreign_request` | bool | Foreign IP request | LOW | Keep |
| `source` | string | Application source | LOW | Keep |
| `session_length_in_minutes` | float | Session duration | LOW | Keep |
| `device_os` | string | Operating system | LOW | Keep |
| `keep_alive_session` | bool | Session kept alive | LOW | Keep |
| `device_distinct_emails_8w` | int | Emails from device | MEDIUM | Keep |
| `device_fraud_count` | int | Fraud count on device | LOW | Keep |
| `month` | int | Month of application | LOW | Keep |
| `fraud_bool` | int | **TARGET** (0/1) | - | Target |

### 3.2 PII Column Classification

```
HIGH RISK (Quasi-Identifiers):
├── income              → Bin: [0-25k, 25-50k, 50-100k, 100k+]
├── customer_age        → Bin: [18-25, 26-35, 36-45, 46-55, 56+]
├── employment_status   → Generalize: {employed, unemployed, other}
└── housing_status      → Generalize: {owned, rented, other}

MEDIUM RISK:
├── name_email_similarity    → Keep (bounded 0-1)
├── zip_count_4w             → Keep (behavioral)
├── date_of_birth_distinct_emails_4w → Keep (aggregated)
├── credit_risk_score        → Keep (model feature)
└── proposed_credit_limit    → Keep (model feature)

LOW RISK:
└── All others → Keep as-is
```

### 3.3 Feature Engineering

```python
# Derived features for Bank Account Fraud
derived_features = {
    "velocity_ratio_6h_24h": "velocity_6h / (velocity_24h + 1e-6)",
    "velocity_ratio_24h_4w": "velocity_24h / (velocity_4w + 1e-6)",
    "address_stability": "current_address_months_count / (prev_address_months_count + 1)",
    "email_risk_score": "date_of_birth_distinct_emails_4w * (1 - name_email_similarity)",
    "device_risk_score": "device_distinct_emails_8w + device_fraud_count * 10",
    "is_high_velocity": "velocity_6h > velocity_6h.quantile(0.95)",
    "is_new_customer": "bank_months_count < 3",
    "credit_request_ratio": "intended_balcon_amount / (proposed_credit_limit + 1)",
}
```

---

## 4. Credit Card Fraud Dataset

### 4.1 Schema

| Column | Type | Description | PII Risk | Action |
|--------|------|-------------|----------|--------|
| `Time` | int | Seconds from first txn | LOW | Keep |
| `V1` - `V28` | float | PCA transformed features | **NONE** | Keep |
| `Amount` | float | Transaction amount | MEDIUM | Keep / Bin |
| `Class` | int | **TARGET** (0/1) | - | Target |

### 4.2 Key Characteristics

```
Dataset Properties:
├── Already anonymized via PCA transformation
├── Original features UNKNOWN (privacy preserved)
├── Time is relative (not absolute timestamp)
├── Highly imbalanced: 0.17% fraud rate
└── Amount is the only interpretable feature

PII Status: ✅ SAFE
├── V1-V28 are PCA components (not reversible)
├── No direct identifiers
├── No quasi-identifiers
└── Can use directly without transformation
```

### 4.3 Feature Engineering

```python
# Derived features for Credit Card
derived_features = {
    "amount_log": "np.log1p(Amount)",
    "amount_zscore": "(Amount - Amount.mean()) / Amount.std()",
    "hour_of_day": "(Time // 3600) % 24",
    "is_night": "hour_of_day.isin([0,1,2,3,4,5,22,23])",
    "is_high_amount": "Amount > Amount.quantile(0.99)",
    "amount_bin": "pd.cut(Amount, bins=[0, 10, 50, 100, 500, 1000, np.inf])",
    # PCA feature interactions
    "v1_v2_ratio": "V1 / (V2 + 1e-6)",
    "v_sum_abs": "np.abs(V1) + np.abs(V2) + ... + np.abs(V28)",
}
```

---

## 5. PaySim Dataset

### 5.1 Schema

| Column | Type | Description | PII Risk | Action |
|--------|------|-------------|----------|--------|
| `step` | int | Time step (1 step = 1 hour) | LOW | Keep |
| `type` | string | Transaction type | LOW | Keep |
| `amount` | float | Transaction amount | MEDIUM | Keep |
| `nameOrig` | string | Sender account ID | **CRITICAL** | **HASH** |
| `oldbalanceOrg` | float | Sender balance before | MEDIUM | Keep |
| `newbalanceOrig` | float | Sender balance after | MEDIUM | Keep |
| `nameDest` | string | Recipient account ID | **CRITICAL** | **HASH** |
| `oldbalanceDest` | float | Recipient balance before | MEDIUM | Keep |
| `newbalanceDest` | float | Recipient balance after | MEDIUM | Keep |
| `isFraud` | int | **TARGET** (0/1) | - | Target |
| `isFlaggedFraud` | int | Flagged by rules | LOW | Keep |

### 5.2 Transaction Types

```
Transaction Types (in order of fraud prevalence):
├── TRANSFER  → High fraud risk (can move to CASH_OUT)
├── CASH_OUT  → High fraud risk (money extraction)
├── PAYMENT   → Low fraud risk
├── CASH_IN   → Low fraud risk
└── DEBIT     → Low fraud risk

Fraud Distribution:
├── TRANSFER: ~0.76% fraud rate
├── CASH_OUT: ~0.18% fraud rate
└── Others: ~0% fraud rate
```

### 5.3 PII Handling

```
CRITICAL PII:
├── nameOrig → SHA-256 hash (one-way)
├── nameDest → SHA-256 hash (one-way)
└── Hashing preserves entity relationships for graph analysis

Implementation:
import hashlib
def hash_account(account_id: str, salt: str = "sentinxfl") -> str:
    return hashlib.sha256(f"{salt}:{account_id}".encode()).hexdigest()[:16]
```

### 5.4 Feature Engineering

```python
# Derived features for PaySim
derived_features = {
    # Balance features
    "balance_change_orig": "oldbalanceOrg - newbalanceOrig",
    "balance_change_dest": "newbalanceDest - oldbalanceDest",
    "balance_ratio_orig": "newbalanceOrig / (oldbalanceOrg + 1)",
    "is_zero_balance_after": "newbalanceOrig == 0",
    "is_full_drain": "(oldbalanceOrg > 0) & (newbalanceOrig == 0)",
    
    # Amount features
    "amount_log": "np.log1p(amount)",
    "amount_to_balance_ratio": "amount / (oldbalanceOrg + 1)",
    "is_exact_balance": "amount == oldbalanceOrg",
    
    # Transaction patterns
    "is_transfer_then_cashout": "type == 'TRANSFER'",  # flagged separately
    "hour_of_day": "step % 24",
    "day_of_week": "(step // 24) % 7",
    "is_night_txn": "hour_of_day.isin([0,1,2,3,4,5])",
    
    # Type encoding
    "type_encoded": "type.map({'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4})",
    
    # Risk indicators
    "fraud_pattern_1": "(type == 'TRANSFER') & is_full_drain",
    "fraud_pattern_2": "(amount > 200000) & (type == 'TRANSFER')",
}
```

---

## 6. Unified Schema

### 6.1 Common Feature Mapping

```python
# Map all datasets to unified schema
unified_schema = {
    "transaction_id": {
        "bank_account": "row_index",  # synthetic
        "credit_card": "row_index",   # synthetic
        "paysim": "hash(nameOrig + step)"
    },
    "timestamp": {
        "bank_account": "month",      # month only
        "credit_card": "Time",        # seconds from start
        "paysim": "step"              # hours from start
    },
    "amount": {
        "bank_account": "intended_balcon_amount",
        "credit_card": "Amount",
        "paysim": "amount"
    },
    "is_fraud": {
        "bank_account": "fraud_bool",
        "credit_card": "Class",
        "paysim": "isFraud"
    },
    "fraud_type": {
        "bank_account": "account_fraud",
        "credit_card": "card_fraud",
        "paysim": "payment_fraud"
    }
}
```

### 6.2 Dataset-Specific Features

```yaml
feature_groups:
  common:
    - amount
    - amount_log
    - is_fraud
    - fraud_type
    
  bank_account_only:
    - income_bin
    - age_bin
    - velocity_6h
    - velocity_24h
    - credit_risk_score
    - device_fraud_count
    
  credit_card_only:
    - V1 through V28
    - hour_of_day
    
  paysim_only:
    - type_encoded
    - balance_change_orig
    - balance_change_dest
    - is_full_drain
```

---

## 7. Data Quality Checks

### 7.1 Validation Rules

```python
validation_rules = {
    "bank_account_fraud": {
        "fraud_bool": {"type": "int", "values": [0, 1]},
        "customer_age": {"type": "int", "min": 18, "max": 100},
        "income": {"type": "float", "min": 0},
        "velocity_6h": {"type": "float", "min": 0},
    },
    "credit_card": {
        "Class": {"type": "int", "values": [0, 1]},
        "Amount": {"type": "float", "min": 0},
        "Time": {"type": "int", "min": 0},
    },
    "paysim": {
        "isFraud": {"type": "int", "values": [0, 1]},
        "type": {"type": "str", "values": ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]},
        "amount": {"type": "float", "min": 0},
    }
}
```

### 7.2 Missing Value Strategy

```yaml
missing_value_handling:
  numeric_columns:
    strategy: "median"
    fallback: 0
    
  categorical_columns:
    strategy: "mode"
    fallback: "UNKNOWN"
    
  critical_columns:
    # If these are missing, drop the row
    - fraud_bool / Class / isFraud  (target)
    - amount / Amount               (key feature)
```

---

## 8. Sampling Strategy

### 8.1 For Development (Fast Iteration)

```python
dev_sample_config = {
    "bank_account_fraud": {
        "sample_size": 100_000,  # 100K from 6M
        "method": "stratified",
        "stratify_column": "fraud_bool"
    },
    "credit_card": {
        "sample_size": 50_000,   # 50K from 285K
        "method": "stratified",
        "stratify_column": "Class"
    },
    "paysim": {
        "sample_size": 100_000,  # 100K from 6.3M
        "method": "stratified",
        "stratify_column": "isFraud"
    }
}
```

### 8.2 For Production (Full Data)

```python
prod_config = {
    "bank_account_fraud": {
        "sample_size": None,  # Full 6M
        "batch_size": 100_000,
        "lazy_loading": True
    },
    "credit_card": {
        "sample_size": None,  # Full 285K
        "batch_size": 50_000,
        "lazy_loading": True
    },
    "paysim": {
        "sample_size": None,  # Full 6.3M
        "batch_size": 100_000,
        "lazy_loading": True
    }
}
```

---

## 9. Train/Test Split Strategy

### 9.1 Temporal Split (Preferred)

```
For datasets with time component:

┌────────────────────────────────────────────────────────────┐
│                    TEMPORAL SPLIT                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ├── Train (70%) ──┼── Val (15%) ──┼── Test (15%) ──┤     │
│  │                 │               │                 │     │
│  │   Earliest      │               │    Latest      │     │
│  │   transactions  │               │    transactions│     │
│  │                 │               │                 │     │
│  └─────────────────┴───────────────┴─────────────────┘     │
│                                                            │
│  NO DATA LEAKAGE: Test set is always in the "future"       │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 9.2 Stratified Split (Backup)

```python
# For datasets without clear temporal ordering
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(sss.split(X, y))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))
```

---

## 10. Class Imbalance Handling

### 10.1 Strategies

```yaml
imbalance_strategies:
  
  training:
    # Option 1: Class weights (preferred)
    class_weight:
      method: "balanced"
      formula: "n_samples / (n_classes * np.bincount(y))"
    
    # Option 2: SMOTE (for small datasets only)
    smote:
      enabled: false  # Memory intensive
      sampling_strategy: 0.5
    
    # Option 3: Undersampling (for very large datasets)
    undersampling:
      enabled: true
      majority_ratio: 10  # 10:1 ratio
  
  evaluation:
    # Never balance test set!
    # Use appropriate metrics instead:
    metrics:
      - AUC-ROC  # Primary (threshold-independent)
      - AUC-PR   # For severe imbalance
      - F1       # At optimal threshold
      - Recall@90Precision  # Business metric
```

### 10.2 Recommended Settings

| Dataset | Fraud Rate | Strategy | Expected AUC |
|---------|------------|----------|--------------|
| Bank Account | 1.4% | Class weights | 0.92+ |
| Credit Card | 0.17% | Class weights + threshold tuning | 0.95+ |
| PaySim | 0.13% | Class weights + TRANSFER/CASH_OUT focus | 0.99+ |

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
