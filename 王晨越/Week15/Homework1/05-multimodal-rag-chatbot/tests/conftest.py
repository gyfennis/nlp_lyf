import os
import sys

import pytest

# Project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("USE_STUB_EMBED", "1")
os.environ.setdefault("USE_STUB_CHAT", "1")
os.environ.setdefault("USE_STUB_VECTOR", "1")
os.environ.setdefault("USE_INMEMORY_QUEUE", "1")
os.environ.setdefault("SQLITE_PATH", os.path.join(ROOT, "tests", "test_db.db"))


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
