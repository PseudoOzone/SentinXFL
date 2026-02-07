# SentinXFL - Security Documentation

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)  
> **Classification**: Internal

---

## 1. Security Overview

### 1.1 Security Principles

| Principle | Implementation |
|-----------|----------------|
| **Defense in Depth** | Multiple security layers (network, application, data) |
| **Least Privilege** | Minimal permissions for each component |
| **Zero Trust** | Verify every request, assume breach |
| **Privacy by Design** | Privacy controls built into architecture |
| **Secure by Default** | Secure configurations out of the box |

### 1.2 Security Scope

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY PERIMETER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Frontend   │    │   Backend   │    │  ML/FL      │         │
│  │  (Next.js)  │◄──►│  (FastAPI)  │◄──►│  Pipeline   │         │
│  │             │    │             │    │             │         │
│  │ • CSP       │    │ • JWT Auth  │    │ • DP Noise  │         │
│  │ • HTTPS     │    │ • RBAC      │    │ • Gradient  │         │
│  │ • XSS Prot  │    │ • Rate Limit│    │   Clipping  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │                  DATA LAYER                      │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │           │
│  │  │  DuckDB  │  │ ChromaDB │  │  Models  │       │           │
│  │  │(Encrypted│  │(Encrypted│  │(Signed)  │       │           │
│  │  │ at rest) │  │ at rest) │  │          │       │           │
│  │  └──────────┘  └──────────┘  └──────────┘       │           │
│  └─────────────────────────────────────────────────┘           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Threat Model

### 2.1 Threat Actors

| Actor | Capability | Motivation | Likelihood |
|-------|------------|------------|------------|
| **External Attacker** | Network access, public endpoints | Data theft, disruption | High |
| **Malicious Insider** | System access, credentials | Data exfiltration | Medium |
| **Compromised Bank** | FL participant, gradient access | Model poisoning | Medium |
| **Nation State** | Advanced persistent threat | Surveillance, disruption | Low |
| **Accidental Insider** | Legitimate access, mistakes | Unintentional breach | High |

### 2.2 Attack Vectors

```yaml
attack_vectors:
  network_attacks:
    - name: "Man-in-the-Middle"
      mitigation: "TLS 1.3 with certificate pinning"
      
    - name: "DDoS"
      mitigation: "Rate limiting, CDN (if deployed)"
      
    - name: "API Abuse"
      mitigation: "Rate limiting, request validation"
      
  application_attacks:
    - name: "SQL Injection"
      mitigation: "Parameterized queries (DuckDB)"
      
    - name: "XSS"
      mitigation: "CSP headers, input sanitization"
      
    - name: "CSRF"
      mitigation: "CSRF tokens, SameSite cookies"
      
    - name: "Broken Authentication"
      mitigation: "JWT with short expiry, refresh tokens"
      
  ml_specific_attacks:
    - name: "Model Inversion"
      mitigation: "Differential privacy (ε=1.0)"
      
    - name: "Membership Inference"
      mitigation: "DP guarantees, output perturbation"
      
    - name: "Model Poisoning"
      mitigation: "Byzantine-robust aggregation (Krum)"
      
    - name: "Gradient Leakage"
      mitigation: "Gradient clipping + DP noise"
      
  data_attacks:
    - name: "Re-identification"
      mitigation: "k-Anonymity (k≥5), PII blocking"
      
    - name: "Linkage Attack"
      mitigation: "Quasi-identifier generalization"
      
    - name: "Data Exfiltration"
      mitigation: "DLP, audit logging, access control"
```

### 2.3 STRIDE Analysis

| Threat | Category | Asset | Mitigation |
|--------|----------|-------|------------|
| Attacker impersonates user | **S**poofing | Authentication | JWT, MFA-ready |
| Attacker modifies data | **T**ampering | Data integrity | Hash-chained audit |
| User denies action | **R**epudiation | Accountability | Immutable audit log |
| PII exposed | **I**nformation Disclosure | Customer data | PII blocking + DP |
| System overwhelmed | **D**enial of Service | Availability | Rate limiting |
| Privilege escalation | **E**levation of Privilege | Access control | RBAC, least privilege |

---

## 3. Authentication & Authorization

### 3.1 JWT Authentication

```python
# JWT configuration
jwt_config = {
    "algorithm": "RS256",  # Asymmetric for better security
    "access_token_expire_minutes": 15,
    "refresh_token_expire_days": 7,
    "issuer": "sentinxfl",
    "audience": "sentinxfl-api"
}

# Token structure
access_token_payload = {
    "sub": "user_id",
    "exp": "expiration_timestamp",
    "iat": "issued_at_timestamp",
    "jti": "unique_token_id",
    "roles": ["analyst", "admin"],
    "permissions": ["read:data", "write:models"],
    "bank_id": "bank_identifier"  # For multi-tenant isolation
}
```

### 3.2 Role-Based Access Control (RBAC)

```yaml
roles:
  viewer:
    description: "Read-only access to dashboards"
    permissions:
      - "read:dashboard"
      - "read:reports"
      
  analyst:
    description: "Fraud analyst operations"
    permissions:
      - "read:dashboard"
      - "read:reports"
      - "read:predictions"
      - "create:reports"
      
  data_scientist:
    description: "ML model operations"
    permissions:
      - "read:dashboard"
      - "read:data"
      - "read:models"
      - "create:models"
      - "update:models"
      
  compliance_officer:
    description: "Compliance and audit access"
    permissions:
      - "read:dashboard"
      - "read:audit"
      - "read:compliance"
      - "create:compliance_reports"
      - "read:pii_certificates"
      
  admin:
    description: "Full system access"
    permissions:
      - "*:*"
      
  fl_participant:
    description: "Federated learning participant"
    permissions:
      - "connect:fl_server"
      - "submit:gradients"
      - "receive:aggregated_model"
```

### 3.3 API Key Management

```python
# For service-to-service communication
api_key_config = {
    "prefix": "sxfl_",
    "length": 32,
    "hash_algorithm": "sha256",
    "rotation_days": 90,
    "max_keys_per_service": 3
}

# Storage: Only hash stored, never plaintext
api_key_record = {
    "key_id": "uuid",
    "key_hash": "sha256_hash",
    "service_name": "bank_a_fl_client",
    "permissions": ["fl:connect", "fl:submit"],
    "created_at": "timestamp",
    "expires_at": "timestamp",
    "last_used": "timestamp",
    "is_active": True
}
```

---

## 4. Data Security

### 4.1 Encryption at Rest

```yaml
encryption_at_rest:
  database:
    engine: "DuckDB"
    encryption: "SQLCipher (AES-256)"
    key_management: "Environment variable (HSM for production)"
    
  vector_store:
    engine: "ChromaDB"
    encryption: "Filesystem encryption (BitLocker/LUKS)"
    
  models:
    format: "Encrypted pickle with signature"
    encryption: "Fernet (AES-128-CBC)"
    signing: "HMAC-SHA256"
    
  audit_logs:
    encryption: "AES-256-GCM"
    integrity: "Hash chain (SHA-256)"
```

### 4.2 Encryption in Transit

```yaml
encryption_in_transit:
  api:
    protocol: "TLS 1.3"
    ciphers:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
    certificate: "Let's Encrypt or internal CA"
    hsts: true
    hsts_max_age: 31536000
    
  fl_communication:
    protocol: "gRPC over TLS"
    mutual_tls: true  # Both client and server authenticate
    certificate_pinning: true
    
  internal:
    protocol: "TLS 1.2+ (localhost exemption for dev)"
```

### 4.3 Key Management

```yaml
key_management:
  development:
    storage: ".env file (gitignored)"
    rotation: "Manual"
    
  production:
    storage: "AWS KMS / Azure Key Vault / HashiCorp Vault"
    rotation: "Automatic (90 days)"
    access: "IAM-controlled"
    
  keys_required:
    - name: "JWT_PRIVATE_KEY"
      type: "RSA-2048"
      purpose: "JWT signing"
      
    - name: "DATABASE_KEY"
      type: "AES-256"
      purpose: "DuckDB encryption"
      
    - name: "MODEL_KEY"
      type: "AES-128"
      purpose: "Model encryption"
      
    - name: "AUDIT_KEY"
      type: "AES-256"
      purpose: "Audit log encryption"
```

---

## 5. ML/FL Security

### 5.1 Gradient Security

```
Gradient Leakage Prevention:

1. Gradient Clipping
   └── Clip gradients to max_norm = 1.0
   └── Prevents outlier gradients from leaking individual data

2. Differential Privacy Noise
   └── Add Gaussian noise: σ = sensitivity × √(2 ln(1.25/δ)) / ε
   └── Parameters: ε=1.0, δ=1e-5
   └── Noise magnitude: ~1.0 per gradient element

3. Secure Aggregation (Future Enhancement)
   └── Clients encrypt gradients
   └── Server only sees sum after decryption
   └── Individual gradients never visible

Implementation:
┌──────────────────────────────────────────────────────────────┐
│  Client Gradient → Clip → Add Noise → Encrypt → Send        │
│                                                              │
│  Server: Aggregate(Decrypt(gradients)) → Update Model       │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Byzantine-Robust Aggregation

```python
# Multi-Krum defense against poisoning
def multi_krum(gradients: List[np.ndarray], f: int, m: int) -> np.ndarray:
    """
    Select m gradients with smallest sum of distances to closest neighbors.
    Tolerates up to f Byzantine (malicious) participants.
    
    Args:
        gradients: List of client gradients
        f: Number of Byzantine participants to tolerate
        m: Number of gradients to select (m ≤ n - f - 2)
    
    Returns:
        Aggregated gradient (average of selected)
    """
    n = len(gradients)
    # Calculate pairwise distances
    distances = compute_pairwise_distances(gradients)
    
    # For each gradient, sum distances to n-f-2 closest neighbors
    scores = []
    for i in range(n):
        sorted_distances = sorted(distances[i])
        score = sum(sorted_distances[:n - f - 2])
        scores.append((score, i))
    
    # Select m gradients with lowest scores
    scores.sort()
    selected_indices = [idx for _, idx in scores[:m]]
    
    # Average selected gradients
    return np.mean([gradients[i] for i in selected_indices], axis=0)
```

### 5.3 Model Integrity

```yaml
model_security:
  signing:
    algorithm: "HMAC-SHA256"
    key: "MODEL_SIGNING_KEY"
    verify_on_load: true
    
  versioning:
    storage: "MLflow model registry"
    immutable: true
    audit_trail: true
    
  access_control:
    read: ["analyst", "data_scientist", "admin"]
    write: ["data_scientist", "admin"]
    deploy: ["admin"]
    
  inference_security:
    input_validation: true
    output_sanitization: true
    rate_limiting: "100 requests/minute/user"
```

---

## 6. Network Security

### 6.1 API Security Headers

```python
# FastAPI security middleware
security_headers = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}
```

### 6.2 Rate Limiting

```yaml
rate_limits:
  default:
    requests: 100
    window: "1 minute"
    
  auth_endpoints:
    requests: 10
    window: "1 minute"
    
  ml_endpoints:
    requests: 20
    window: "1 minute"
    
  report_generation:
    requests: 5
    window: "1 minute"
    
  fl_endpoints:
    requests: 1000  # Higher for FL communication
    window: "1 minute"
    
  penalty:
    block_duration: "15 minutes"
    after_violations: 3
```

### 6.3 Input Validation

```python
# Pydantic validation example
from pydantic import BaseModel, Field, validator
import re

class DatasetLoadRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=500)
    dataset_type: Literal["bank_account", "credit_card", "paysim"]
    
    @validator("path")
    def validate_path(cls, v):
        # Prevent path traversal
        if ".." in v or v.startswith("/"):
            raise ValueError("Invalid path")
        # Allow only specific characters
        if not re.match(r'^[a-zA-Z0-9_\-./:\\]+$', v):
            raise ValueError("Path contains invalid characters")
        return v

class PredictionRequest(BaseModel):
    model_id: str = Field(..., regex=r'^[a-f0-9-]{36}$')
    data: List[Dict[str, Any]] = Field(..., max_items=1000)
    
    @validator("data", each_item=True)
    def validate_data_item(cls, v):
        # Check for suspicious patterns
        for key, value in v.items():
            if isinstance(value, str) and len(value) > 1000:
                raise ValueError(f"Field {key} exceeds max length")
        return v
```

---

## 7. Logging & Monitoring

### 7.1 Security Logging

```yaml
security_logs:
  authentication:
    events:
      - "login_success"
      - "login_failure"
      - "logout"
      - "token_refresh"
      - "password_change"
    retention: "1 year"
    
  authorization:
    events:
      - "access_granted"
      - "access_denied"
      - "permission_change"
    retention: "1 year"
    
  data_access:
    events:
      - "dataset_load"
      - "pii_detection"
      - "data_export"
      - "model_training"
      - "prediction_request"
    retention: "2 years"
    
  system:
    events:
      - "service_start"
      - "service_stop"
      - "config_change"
      - "error"
    retention: "90 days"
```

### 7.2 Intrusion Detection

```yaml
intrusion_detection:
  rules:
    - name: "Brute Force Login"
      condition: "login_failures > 5 in 5 minutes"
      action: "block_ip + alert"
      
    - name: "API Abuse"
      condition: "requests > 1000 in 1 minute"
      action: "rate_limit + alert"
      
    - name: "Data Exfiltration"
      condition: "export_requests > 10 in 1 hour"
      action: "alert + require_approval"
      
    - name: "Unusual Hours Access"
      condition: "admin_access between 2am-5am"
      action: "alert"
      
    - name: "Privilege Escalation Attempt"
      condition: "access_denied to admin_endpoint > 3"
      action: "block_user + alert"
```

### 7.3 Audit Trail Integrity

```python
# Hash-chained audit log
class AuditEntry:
    def __init__(self, action: str, details: dict, previous_hash: str):
        self.timestamp = datetime.utcnow().isoformat()
        self.action = action
        self.details = details
        self.previous_hash = previous_hash
        self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        data = f"{self.timestamp}|{self.action}|{json.dumps(self.details)}|{self.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    @staticmethod
    def verify_chain(entries: List['AuditEntry']) -> bool:
        """Verify audit log integrity"""
        for i in range(1, len(entries)):
            if entries[i].previous_hash != entries[i-1].entry_hash:
                return False
            if entries[i]._compute_hash() != entries[i].entry_hash:
                return False
        return True
```

---

## 8. Secure Development

### 8.1 Dependency Security

```yaml
dependency_security:
  scanning:
    tool: "pip-audit, safety"
    frequency: "weekly + on commit"
    
  policy:
    severity_block: "high, critical"
    auto_update: "patch versions only"
    review_required: "minor, major versions"
    
  lockfile:
    use: true
    tool: "pip-compile (pip-tools)"
```

### 8.2 Code Security

```yaml
code_security:
  static_analysis:
    tool: "bandit, semgrep"
    run_on: "pre-commit, CI"
    
  secrets_detection:
    tool: "detect-secrets, gitleaks"
    run_on: "pre-commit, CI"
    
  code_review:
    required: true
    security_checklist:
      - "No hardcoded credentials"
      - "Input validation on all endpoints"
      - "Output encoding for user data"
      - "Parameterized queries"
      - "Proper error handling (no stack traces to users)"
```

### 8.3 Secure Coding Guidelines

```
OWASP Top 10 Mitigations:

1. Injection
   └── Use parameterized queries (DuckDB)
   └── Validate all inputs with Pydantic

2. Broken Authentication
   └── JWT with short expiry
   └── Secure password hashing (argon2)
   └── Rate limiting on auth endpoints

3. Sensitive Data Exposure
   └── TLS everywhere
   └── Encryption at rest
   └── PII blocking pipeline

4. XML External Entities
   └── N/A (no XML processing)

5. Broken Access Control
   └── RBAC with permission checks
   └── Principle of least privilege

6. Security Misconfiguration
   └── Secure defaults
   └── No debug in production
   └── Security headers

7. Cross-Site Scripting
   └── CSP headers
   └── Input sanitization
   └── React auto-escaping

8. Insecure Deserialization
   └── JSON only (no pickle from users)
   └── Schema validation

9. Using Components with Vulnerabilities
   └── Regular dependency updates
   └── Automated scanning

10. Insufficient Logging
    └── Comprehensive audit logging
    └── Log integrity verification
```

---

## 9. Incident Response

### 9.1 Security Incident Categories

| Category | Severity | Response Time | Example |
|----------|----------|---------------|---------|
| Critical | P1 | 15 minutes | Data breach, system compromise |
| High | P2 | 1 hour | Authentication bypass, DoS |
| Medium | P3 | 4 hours | Unauthorized access attempt |
| Low | P4 | 24 hours | Policy violation |

### 9.2 Incident Response Steps

```yaml
incident_response:
  1_detection:
    - "Automated alerts trigger"
    - "User reports"
    - "Monitoring anomalies"
    
  2_triage:
    - "Assess severity"
    - "Identify scope"
    - "Notify stakeholders"
    
  3_containment:
    - "Isolate affected systems"
    - "Block suspicious IPs"
    - "Revoke compromised credentials"
    
  4_eradication:
    - "Remove malware/backdoors"
    - "Patch vulnerabilities"
    - "Reset credentials"
    
  5_recovery:
    - "Restore from clean backup"
    - "Verify system integrity"
    - "Resume operations"
    
  6_lessons_learned:
    - "Post-incident review"
    - "Update procedures"
    - "Improve defenses"
```

---

## 10. Security Checklist

### 10.1 Development Checklist

- [ ] No hardcoded secrets in code
- [ ] All inputs validated
- [ ] Parameterized database queries
- [ ] Error messages don't leak info
- [ ] Logging excludes sensitive data
- [ ] Dependencies scanned for vulnerabilities
- [ ] Security headers configured
- [ ] HTTPS enforced
- [ ] Rate limiting enabled
- [ ] Authentication on all protected endpoints

### 10.2 Deployment Checklist

- [ ] Debug mode disabled
- [ ] Secrets in environment/vault
- [ ] TLS certificates valid
- [ ] Firewall rules configured
- [ ] Monitoring enabled
- [ ] Backup procedures tested
- [ ] Incident response plan ready
- [ ] Access controls verified
- [ ] Security scan passed

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
