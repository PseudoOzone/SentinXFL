# SentinXFL

Privacy-aware federated fraud-detection research platform built as a final-year engineering project.

SentinXFL explores how financial institutions can train and evaluate fraud models across separate clients without treating a centralized raw-data pool as the default architecture. The repository combines data validation, PII-risk screening, classical and deep-learning models, federated aggregation, differential-privacy experiments, adversarial-client simulation, explainability, an API, and a React dashboard.

> **Project status:** academic research prototype. It is not a certified banking product, a compliance assessment, or a production-ready security control.

## What this repository demonstrates

- Client-level and centralized fraud-model experiments
- Federated training workflows using Flower
- FedAvg, coordinate median, trimmed mean, and Multi-Krum-style robust aggregation experiments
- Differential-privacy components and privacy-accounting experiments
- Label-flipping and poisoning simulations
- PII-risk detection using column, pattern, uniqueness, and entropy-based checks
- Evidence-oriented run artifacts, metrics, reports, and plots
- FastAPI endpoints and a React/TypeScript dashboard
- Automated linting, tests, frontend builds, CodeQL analysis, dependency review, and SBOM generation through GitHub Actions

## Architecture

```text
Client datasets
      |
      v
Validation and PII-risk screening
      |
      v
Local preprocessing and model training
      |
      v
Federated aggregation and attack simulation
      |
      v
Evaluation, privacy reports, explanations, and run artifacts
      |
      +--> FastAPI API
      +--> React dashboard
```

## Repository structure

```text
SentinXFL/
├── src/sentinxfl/
│   ├── api/            # FastAPI application and routes
│   ├── core/           # Configuration and logging
│   ├── data/           # Dataset loading and partitioning
│   ├── privacy/        # PII-risk checks and privacy utilities
│   ├── ml/             # Model training and evaluation
│   ├── fl/             # Federated clients, server, and aggregators
│   ├── intelligence/   # Pattern and reporting experiments
│   └── llm/            # Local explanation and RAG experiments
├── dashboard/          # React and TypeScript interface
├── tests/              # Backend and integration tests
├── knowledge/          # Design and testing documentation
├── data/               # Local datasets and generated data
└── .github/workflows/  # CI and security automation
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- 8 GB RAM or more
- An NVIDIA GPU is optional; several workflows can run on CPU
- Ollama is optional for local explanation features

### Backend

```bash
git clone https://github.com/PseudoOzone/SentinXFL.git
cd SentinXFL

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env

uvicorn sentinxfl.api.app:app --reload --port 8000
```

API documentation is available at `http://localhost:8000/docs` while the server is running.

### Dashboard

```bash
cd dashboard
npm ci
npm run dev
```

The development dashboard normally starts at `http://localhost:5173`.

### CLI

```bash
sentinxfl info
sentinxfl scan --dataset all
```

Run `sentinxfl --help` for the commands available in the installed version.

## Testing and quality checks

```bash
pytest tests/
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/sentinxfl
```

The GitHub Actions workflow also builds the frontend and runs CodeQL. Test totals may change as the project evolves; use the current workflow result rather than a hard-coded badge as the source of truth.

## Security and privacy scope

SentinXFL includes security-oriented design experiments, but those experiments do not by themselves establish GDPR, DPDPA, RBI, PCI-DSS, or any other regulatory compliance.

Before any real deployment, the following prototype components must be replaced or hardened:

- in-memory users, sessions, tokens, and upload metadata
- demo authentication and seeded accounts
- public self-registration and role assignment
- local file storage and extension-based upload validation
- single-process rate limiting
- development CORS and server settings
- unreviewed model, privacy, and LLM configurations

Use synthetic or properly authorized datasets only. Do not upload real financial or personal data to an unreviewed deployment.

## Important implementation boundaries

- Raw datasets remain local to simulated clients, but the current repository does **not** claim cryptographic secure aggregation in which the server is unable to inspect individual client updates.
- Transport encryption and federated learning are not substitutes for secure aggregation, differential privacy, authentication, or update validation.
- Differential-privacy values are experiment configurations. An epsilon value is meaningful only with its mechanism, delta, clipping bound, sampling process, number of steps, and accounting method.
- The DP-SGD and RDP-accounting modules are educational implementations and should be validated against established libraries such as Opacus before any formal privacy claim.
- Robust aggregation can reduce the effect of some outlier or poisoning strategies, but its guarantees depend on assumptions about client count, attacker capability, data heterogeneity, and update geometry.

## Experimental limitations

- Results depend on dataset composition, client partitioning, random seeds, and model configuration.
- Heuristic PII-risk detection can produce both false positives and false negatives.
- Differential privacy requires mechanism-specific accounting and does not automatically apply to every workflow in the repository.
- Robust aggregation reduces some attack effects but does not guarantee Byzantine security under every threat model.
- LLM-generated explanations are supporting artifacts, not authoritative fraud decisions.
- No patent status or regulatory certification is claimed by this repository.

## Academic context

Developed at SRM Institute of Science and Technology under academic supervision.

- **Lead developer:** Anshuman Bakshi
- **Contributor:** Komal
- **Supervisor:** Dr. Kiruthika

## License

Proprietary academic-use repository. See the repository license terms before reuse.
