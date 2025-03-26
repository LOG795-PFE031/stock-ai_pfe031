import pytest
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment variables."""
    os.environ["RABBITMQ_HOST"] = "test-rabbitmq"
    os.environ["RABBITMQ_PORT"] = "5672"
    os.environ["RABBITMQ_USER"] = "test-user"
    os.environ["RABBITMQ_PASS"] = "test-pass"
    os.environ["API_HOST"] = "0.0.0.0"
    os.environ["API_PORT"] = "8092"
    os.environ["RATE_LIMIT"] = "2"
    os.environ["RATE_LIMIT_PERIOD"] = "1"
    yield
    # Clean up environment variables after tests
    for key in ["RABBITMQ_HOST", "RABBITMQ_PORT", "RABBITMQ_USER", "RABBITMQ_PASS",
                "API_HOST", "API_PORT", "RATE_LIMIT", "RATE_LIMIT_PERIOD"]:
        os.environ.pop(key, None) 