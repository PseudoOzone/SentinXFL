"""
Pytest configuration and fixtures for SentinXFL.
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Get data directory."""
    return project_root / "data" / "datasets"


@pytest.fixture(scope="session")
def test_data_available(data_dir):
    """Check if test data is available."""
    return (data_dir / "Base.csv").exists()
