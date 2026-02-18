# Security Policy — SentinXFL

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | ✅ Active |
| < 2.0   | ❌ EOL    |

## Reporting a Vulnerability

**Do NOT open a public GitHub Issue for security vulnerabilities.**

1. Email: **anshuman.bakshi@srmist.edu.in**
2. Subject: `[SECURITY] SentinXFL — <brief description>`
3. Include: steps to reproduce, impact assessment, affected component.
4. Expect acknowledgement within **48 hours** and a fix target within **7 business days**.

## Security Architecture Summary

### Authentication
- **Method**: Bearer token (in-memory store; production should migrate to signed JWT + DB).
- **Password hashing**: bcrypt via `passlib`.
- **Login rate limiting**: 5 attempts per email per 5-minute window.
- **Default credentials**: `admin@sentinxfl.com` / `admin123` — **CHANGE IN PRODUCTION**.

### Authorization
- **RBAC**: Two roles — `client` (bank user) and `employee` (SentinXFL admin).
- **Bank isolation**: Clients can only access their own bank's data/uploads.

### Transport & Headers
- CORS restricted to configured origins (no wildcard).
- Security headers: `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Permissions-Policy`.
- Request body size limit: 10 MB (except uploads).
- Correlation ID (`X-Request-ID`) on every response.

### Data Protection
- **PII**: 5-Gate blocking pipeline with differential privacy (ε = 1.0 default).
- **Federated Learning**: Byzantine-robust aggregation, gradient clipping.
- **Logs**: PII and credential redaction applied automatically.

### Secret Management
- `SECRET_KEY` validated at startup — weak/default values rejected outside `development` environment.
- `.env` excluded from version control via `.gitignore`.
- `detect-secrets` pre-commit hook available.

## Threat Model

| Threat | Mitigation |
|--------|------------|
| Credential stuffing | Login rate limiting, bcrypt cost factor |
| Path traversal (upload) | Filename sanitization + `is_relative_to()` check |
| CORS abuse | Restricted origin list from config |
| Error oracle | Generic 500 messages; details only in server logs |
| Model poisoning (FL) | Byzantine-robust aggregation (Multi-Krum, Trimmed Mean) |
| Privacy leakage | Differential privacy with RDP accounting |

## Hardening Checklist (Deployment)

- [ ] Set `SECRET_KEY` to a strong random value
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure `CORS_ORIGINS` to your frontend domain only
- [ ] Enable HTTPS / TLS termination at reverse proxy
- [ ] Change default user passwords
- [ ] Mount `data/` and `logs/` on encrypted storage
- [ ] Set up log aggregation and alerting
- [ ] Run `pre-commit install` for all contributors
