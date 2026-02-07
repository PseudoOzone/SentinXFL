# SentinXFL - Testing Strategy

> **Version**: 2.0  
> **Last Updated**: February 5, 2026  
> **Author**: Anshuman Bakshi (RA2211033010117)

---

## 1. Testing Philosophy

### 1.1 Testing Pyramid

```
                    ┌─────────────────┐
                    │   E2E Tests     │  5%   (slow, brittle)
                    │   (Playwright)  │
                    ├─────────────────┤
                    │  Integration    │  20%  (medium speed)
                    │  Tests          │
                    ├─────────────────┤
                    │                 │
                    │   Unit Tests    │  75%  (fast, isolated)
                    │   (pytest)      │
                    │                 │
                    └─────────────────┘
```

### 1.2 Coverage Targets

| Module | Target Coverage | Priority |
|--------|-----------------|----------|
| PII Pipeline | 95%+ | P0 (Patent core) |
| DP Mechanisms | 90%+ | P0 (Privacy critical) |
| ML Models | 85%+ | P1 |
| FL System | 85%+ | P1 |
| API Endpoints | 90%+ | P1 |
| LLM/RAG | 70%+ | P2 (Hard to test) |
| Frontend | 60%+ | P2 |

---

## 2. Unit Testing

### 2.1 Framework & Tools

```yaml
unit_testing:
  framework: "pytest"
  assertion_library: "pytest (native)"
  mocking: "pytest-mock, unittest.mock"
  fixtures: "pytest fixtures, conftest.py"
  parametrization: "pytest.mark.parametrize"
  coverage: "pytest-cov"
  
  plugins:
    - pytest-asyncio     # Async test support
    - pytest-xdist       # Parallel execution
    - pytest-randomly    # Randomize test order
    - hypothesis         # Property-based testing
```

### 2.2 Unit Test Structure

```python
# tests/unit/test_pii_detector.py
import pytest
from sentinxfl.pii.detector import PIIDetector

class TestPIIDetector:
    """Test suite for PII detection module"""
    
    @pytest.fixture
    def detector(self):
        return PIIDetector()
    
    @pytest.fixture
    def sample_data(self):
        return {
            "high_entropy": ["a1b2c3d4", "x9y8z7w6", ...],  # Random IDs
            "low_entropy": [1, 1, 2, 2, 3, 3, ...],          # Repeated values
            "email_column": ["test@example.com", ...],
            "numeric_column": [100.0, 200.0, 300.0, ...]
        }
    
    # Test entropy calculation
    def test_entropy_calculation_high(self, detector, sample_data):
        """High entropy columns should be flagged"""
        entropy = detector.calculate_entropy(sample_data["high_entropy"])
        assert entropy > 4.0, "High entropy data should have entropy > 4.0"
    
    def test_entropy_calculation_low(self, detector, sample_data):
        """Low entropy columns should not be flagged"""
        entropy = detector.calculate_entropy(sample_data["low_entropy"])
        assert entropy < 4.0, "Low entropy data should have entropy < 4.0"
    
    # Test pattern matching
    @pytest.mark.parametrize("email,expected", [
        ("test@example.com", True),
        ("user.name@domain.co.uk", True),
        ("invalid-email", False),
        ("@nodomain.com", False),
    ])
    def test_email_pattern(self, detector, email, expected):
        """Email pattern should match valid emails"""
        assert detector.is_email(email) == expected
    
    # Test edge cases
    def test_empty_column(self, detector):
        """Empty columns should not crash"""
        result = detector.detect_pii([])
        assert result.pii_detected == False
    
    def test_null_values(self, detector):
        """Null values should be handled"""
        result = detector.detect_pii([None, None, "value"])
        assert result is not None
```

### 2.3 Test Organization

```
tests/
├── unit/
│   ├── conftest.py           # Shared fixtures
│   ├── test_data_loader.py
│   ├── test_data_splitter.py
│   ├── pii/
│   │   ├── test_detector.py
│   │   ├── test_transformer.py
│   │   ├── test_certifier.py
│   │   └── test_audit.py
│   ├── models/
│   │   ├── test_xgboost.py
│   │   ├── test_lightgbm.py
│   │   ├── test_tabnet.py
│   │   └── test_ensemble.py
│   ├── dp/
│   │   ├── test_mechanisms.py
│   │   ├── test_accountant.py
│   │   └── test_budget.py
│   ├── fl/
│   │   ├── test_server.py
│   │   ├── test_client.py
│   │   └── test_aggregators.py
│   └── llm/
│       ├── test_generator.py
│       └── test_guards.py
├── integration/
│   ├── test_pipeline.py
│   ├── test_api.py
│   └── test_fl_simulation.py
└── e2e/
    ├── test_full_flow.py
    └── test_dashboard.py
```

---

## 3. Integration Testing

### 3.1 API Integration Tests

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from sentinxfl.main import app

class TestDataAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def loaded_dataset(self, client):
        """Fixture that loads a test dataset"""
        response = client.post("/api/v1/data/datasets/load", json={
            "path": "tests/fixtures/sample_data.csv",
            "dataset_type": "bank_account"
        })
        assert response.status_code == 200
        return response.json()["dataset_id"]
    
    def test_load_dataset(self, client):
        """Test dataset loading endpoint"""
        response = client.post("/api/v1/data/datasets/load", json={
            "path": "tests/fixtures/sample_data.csv",
            "dataset_type": "bank_account"
        })
        assert response.status_code == 200
        assert "dataset_id" in response.json()
    
    def test_get_schema(self, client, loaded_dataset):
        """Test schema retrieval after loading"""
        response = client.get(f"/api/v1/data/datasets/{loaded_dataset}/schema")
        assert response.status_code == 200
        schema = response.json()
        assert "columns" in schema
        assert "pii_columns" in schema
    
    def test_split_dataset(self, client, loaded_dataset):
        """Test temporal split creation"""
        response = client.post(f"/api/v1/data/datasets/{loaded_dataset}/split", json={
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "method": "temporal"
        })
        assert response.status_code == 200
        split = response.json()
        assert split["train_size"] + split["val_size"] + split["test_size"] > 0
```

### 3.2 PII Pipeline Integration

```python
# tests/integration/test_pii_pipeline.py
import pytest
from sentinxfl.pii import PIIPipeline

class TestPIIPipelineIntegration:
    """Integration tests for complete PII blocking pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        return PIIPipeline(config={
            "entropy_threshold": 4.0,
            "cardinality_threshold": 0.9,
            "k_anonymity_min": 5
        })
    
    @pytest.fixture
    def test_dataframe(self):
        """DataFrame with known PII columns"""
        import polars as pl
        return pl.DataFrame({
            "customer_id": [f"CUST{i:06d}" for i in range(100)],  # PII
            "email": [f"user{i}@example.com" for i in range(100)],  # PII
            "age": [25, 30, 35, 40, 45] * 20,  # QID
            "amount": [100.0 + i for i in range(100)],  # Safe
            "is_fraud": [0] * 95 + [1] * 5  # Target
        })
    
    def test_full_pipeline(self, pipeline, test_dataframe):
        """Test complete pipeline execution"""
        result = pipeline.process(test_dataframe)
        
        # Verify PII detected
        assert "customer_id" in result.pii_columns
        assert "email" in result.pii_columns
        
        # Verify certificate generated
        assert result.certificate is not None
        assert result.certificate.status == "certified"
        
        # Verify no PII in output
        for col in result.pii_columns:
            assert col not in result.safe_data.columns
    
    def test_blocking_gate(self, pipeline, test_dataframe):
        """Ensure hard blocking prevents any PII leakage"""
        result = pipeline.process(test_dataframe)
        
        # Check each output column for PII patterns
        for col in result.safe_data.columns:
            values = result.safe_data[col].to_list()
            for val in values:
                if isinstance(val, str):
                    # No emails
                    assert "@" not in val
                    # No sequential IDs
                    assert not val.startswith("CUST")
```

### 3.3 FL Simulation Integration

```python
# tests/integration/test_fl_simulation.py
import pytest
from sentinxfl.fl import FLSimulator

class TestFLSimulation:
    @pytest.fixture
    def simulator(self):
        return FLSimulator(
            num_clients=3,
            num_rounds=5,
            aggregation="fedavg"
        )
    
    def test_fl_convergence(self, simulator, test_dataset):
        """FL training should converge"""
        result = simulator.run(test_dataset)
        
        # Model should improve over rounds
        assert result.final_metrics["auc_roc"] > result.initial_metrics["auc_roc"]
        
        # Convergence within 5 rounds
        assert result.rounds_completed <= 5
    
    def test_byzantine_robustness(self, simulator, test_dataset):
        """FL should tolerate Byzantine clients"""
        simulator.add_byzantine_client(attack_type="random")
        result = simulator.run(test_dataset)
        
        # Should still converge
        assert result.final_metrics["auc_roc"] > 0.7
        
        # Byzantine client should be detected
        assert len(result.flagged_clients) >= 1
```

---

## 4. End-to-End Testing

### 4.1 E2E Framework

```yaml
e2e_testing:
  framework: "Playwright (Python)"
  browser: "Chromium (headless)"
  
  setup:
    - "Start backend (FastAPI)"
    - "Start frontend (Next.js)"
    - "Initialize test database"
    - "Load test datasets"
```

### 4.2 E2E Test Example

```python
# tests/e2e/test_dashboard.py
import pytest
from playwright.sync_api import Page, expect

class TestDashboard:
    @pytest.fixture
    def page(self, browser):
        page = browser.new_page()
        page.goto("http://localhost:3000")
        return page
    
    def test_executive_overview_loads(self, page: Page):
        """Executive Overview page should load with data"""
        page.goto("http://localhost:3000/dashboard")
        
        # Check page loaded
        expect(page).to_have_title("SentinXFL - Executive Overview")
        
        # Check KPI cards present
        expect(page.locator("[data-testid='kpi-fraud-detected']")).to_be_visible()
        expect(page.locator("[data-testid='kpi-risk-score']")).to_be_visible()
        
        # Check charts rendered
        expect(page.locator("[data-testid='risk-distribution-chart']")).to_be_visible()
    
    def test_generate_report(self, page: Page):
        """Should generate fraud report"""
        page.goto("http://localhost:3000/export")
        
        # Fill form
        page.select_option("[data-testid='report-type']", "executive_summary")
        page.click("[data-testid='generate-button']")
        
        # Wait for generation
        expect(page.locator("[data-testid='report-ready']")).to_be_visible(timeout=30000)
        
        # Download should be available
        expect(page.locator("[data-testid='download-button']")).to_be_enabled()
```

### 4.3 Full Flow E2E

```python
# tests/e2e/test_full_flow.py
import pytest

class TestFullFlow:
    """Test complete user journey"""
    
    def test_fraud_detection_flow(self, page, api_client):
        """
        Complete flow:
        1. Load dataset
        2. Train model
        3. Run predictions
        4. View dashboard
        5. Generate report
        """
        # 1. Load dataset via API
        dataset_id = api_client.load_dataset("bank_account", "test_data.csv")
        
        # 2. Train model
        model_id = api_client.train_model(dataset_id, "ensemble")
        api_client.wait_for_training(model_id, timeout=300)
        
        # 3. Run predictions
        predictions = api_client.predict(model_id, test_transactions)
        assert len(predictions) == len(test_transactions)
        
        # 4. View dashboard
        page.goto("http://localhost:3000/dashboard")
        expect(page.locator("[data-testid='kpi-fraud-detected']")).to_contain_text(str(sum(predictions)))
        
        # 5. Generate report
        page.goto("http://localhost:3000/export")
        page.click("[data-testid='generate-button']")
        expect(page.locator("[data-testid='report-ready']")).to_be_visible(timeout=60000)
```

---

## 5. Specialized Testing

### 5.1 Property-Based Testing (Hypothesis)

```python
# tests/unit/test_pii_properties.py
from hypothesis import given, strategies as st, assume
from sentinxfl.pii import PIIDetector

class TestPIIProperties:
    """Property-based tests for PII detection"""
    
    @given(st.lists(st.integers(), min_size=10))
    def test_entropy_non_negative(self, data):
        """Entropy should always be non-negative"""
        detector = PIIDetector()
        entropy = detector.calculate_entropy(data)
        assert entropy >= 0
    
    @given(st.lists(st.text(min_size=1), min_size=10))
    def test_detector_never_crashes(self, data):
        """Detector should handle any input"""
        detector = PIIDetector()
        try:
            result = detector.detect_pii(data)
            assert result is not None
        except Exception as e:
            pytest.fail(f"Detector crashed with: {e}")
    
    @given(st.emails())
    def test_all_emails_detected(self, email):
        """All valid emails should be detected as PII"""
        detector = PIIDetector()
        assert detector.is_email(email) == True
```

### 5.2 Privacy Testing

```python
# tests/unit/test_dp_privacy.py
import pytest
import numpy as np
from sentinxfl.dp import GaussianMechanism, RDPAccountant

class TestDifferentialPrivacy:
    """Tests verifying DP guarantees"""
    
    def test_noise_calibration(self):
        """Verify noise magnitude matches theoretical bounds"""
        mechanism = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        
        # Generate many samples
        samples = [mechanism.add_noise(0.0) for _ in range(10000)]
        
        # Standard deviation should match theoretical
        expected_sigma = mechanism.sigma
        actual_sigma = np.std(samples)
        
        # Allow 10% tolerance
        assert abs(actual_sigma - expected_sigma) / expected_sigma < 0.1
    
    def test_budget_composition(self):
        """RDP composition should be correct"""
        accountant = RDPAccountant()
        
        # Apply mechanism 10 times
        epsilon_single = 0.1
        for _ in range(10):
            accountant.step(epsilon=epsilon_single, delta=1e-6)
        
        # Get final (ε,δ)-DP guarantee
        epsilon_total, delta_total = accountant.get_privacy_spent(target_delta=1e-5)
        
        # Should be less than naive composition (10 * 0.1 = 1.0)
        assert epsilon_total < 1.0
        
        # But not too tight (sanity check)
        assert epsilon_total > 0.3
    
    def test_gradient_clipping(self):
        """Gradient clipping should bound sensitivity"""
        from sentinxfl.dp import clip_gradients
        
        gradients = [np.array([100.0, 200.0, -300.0])]
        max_norm = 1.0
        
        clipped = clip_gradients(gradients, max_norm)
        
        # Each gradient should have norm <= max_norm
        for grad in clipped:
            assert np.linalg.norm(grad) <= max_norm + 1e-6
```

### 5.3 Performance Testing

```python
# tests/performance/test_inference_latency.py
import pytest
import time
from statistics import mean, stdev

class TestInferenceLatency:
    """Performance tests for inference"""
    
    @pytest.fixture
    def model(self, trained_model_path):
        return load_model(trained_model_path)
    
    def test_single_prediction_latency(self, model, single_transaction):
        """Single prediction should be < 50ms"""
        times = []
        for _ in range(100):
            start = time.perf_counter()
            model.predict(single_transaction)
            times.append(time.perf_counter() - start)
        
        p95 = sorted(times)[95]
        assert p95 < 0.050, f"P95 latency {p95*1000:.1f}ms exceeds 50ms"
    
    def test_batch_prediction_throughput(self, model, batch_transactions):
        """Batch of 1000 should complete in < 1s"""
        start = time.perf_counter()
        model.predict(batch_transactions)  # 1000 transactions
        elapsed = time.perf_counter() - start
        
        throughput = len(batch_transactions) / elapsed
        assert throughput > 1000, f"Throughput {throughput:.0f}/s is too low"
    
    @pytest.mark.slow
    def test_memory_usage(self, model, large_batch):
        """Memory usage should stay bounded"""
        import tracemalloc
        
        tracemalloc.start()
        for _ in range(100):
            model.predict(large_batch)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Peak memory should be < 2GB
        assert peak < 2 * 1024 * 1024 * 1024
```

---

## 6. Test Configuration

### 6.1 pytest.ini

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    gpu: marks tests that require GPU

# Coverage
addopts = 
    --cov=src/sentinxfl
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# Async
asyncio_mode = auto

# Timeout
timeout = 60
```

### 6.2 conftest.py

```python
# tests/conftest.py
import pytest
import polars as pl
from pathlib import Path

# Fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def test_data_dir():
    return FIXTURES_DIR

@pytest.fixture(scope="session")
def sample_bank_account_data():
    """Sample Bank Account Fraud data for testing"""
    return pl.read_csv(FIXTURES_DIR / "sample_bank_account.csv")

@pytest.fixture(scope="session")
def sample_credit_card_data():
    """Sample Credit Card data for testing"""
    return pl.read_csv(FIXTURES_DIR / "sample_credit_card.csv")

@pytest.fixture(scope="function")
def clean_database(tmp_path):
    """Fresh database for each test"""
    db_path = tmp_path / "test.duckdb"
    yield str(db_path)
    # Cleanup after test
    if db_path.exists():
        db_path.unlink()

@pytest.fixture
def mock_llm(mocker):
    """Mock LLM for faster tests"""
    mock = mocker.patch("sentinxfl.llm.model.generate")
    mock.return_value = "Mock generated text"
    return mock
```

---

## 7. CI/CD Integration

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      
      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Run integration tests
        run: pytest tests/integration -v
  
  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
      
      - name: Install backend
        run: pip install -e ".[dev]"
      
      - name: Install frontend
        run: cd frontend && npm ci
      
      - name: Install Playwright
        run: playwright install chromium
      
      - name: Start services
        run: |
          uvicorn src.sentinxfl.main:app &
          cd frontend && npm run dev &
          sleep 10
      
      - name: Run E2E tests
        run: pytest tests/e2e -v
```

---

## 8. Test Data Management

### 8.1 Fixtures

```
tests/fixtures/
├── sample_bank_account.csv      # 1000 rows subset
├── sample_credit_card.csv       # 1000 rows subset
├── sample_paysim.csv            # 1000 rows subset
├── pii_test_data.csv            # Data with known PII
├── clean_test_data.csv          # Data without PII
├── edge_cases.csv               # Edge cases
└── mock_responses/
    ├── llm_responses.json
    └── api_responses.json
```

### 8.2 Data Generation

```python
# tests/fixtures/generate_fixtures.py
"""Generate test fixtures from real datasets"""
import polars as pl

def generate_sample(source_path: str, output_path: str, n_rows: int = 1000):
    """Generate stratified sample for testing"""
    df = pl.scan_csv(source_path)
    
    # Stratified sample
    fraud = df.filter(pl.col("is_fraud") == 1).head(int(n_rows * 0.1))
    normal = df.filter(pl.col("is_fraud") == 0).head(int(n_rows * 0.9))
    
    sample = pl.concat([fraud, normal]).collect()
    sample.write_csv(output_path)
    
    print(f"Generated {len(sample)} rows to {output_path}")
```

---

*Document Version: 2.0 | Author: Anshuman Bakshi | Date: February 5, 2026*
