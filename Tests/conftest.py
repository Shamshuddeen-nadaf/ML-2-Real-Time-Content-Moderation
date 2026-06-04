import pytest
from fastapi.testclient import TestClient
from API.app import app

@pytest.fixture
def client():
  return TestClient(app)


