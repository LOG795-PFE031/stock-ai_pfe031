import sys
import os
import pytest
from fastapi.testclient import TestClient

# Ajoute le dossier racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from services.data_ingestion.main import app

@pytest.fixture(scope="module")
def test_client():
    """Fixture for FastAPI test client."""
    return TestClient(app)
