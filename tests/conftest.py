"""
Shared fixtures for API integration tests.

These tests hit the live serving stack via Nginx (http://localhost) and the
PostgreSQL container (localhost:5432). Run `docker compose up -d` first.
"""

import os
from collections.abc import Iterator

import pytest
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

BASE_URL = os.getenv("MLOPS_TEST_BASE_URL", "http://localhost")
DATABASE_URL = os.getenv(
    "MLOPS_TEST_DATABASE_URL",
    "postgresql://mlops:mlops@localhost:5432/mlops",
)


# ── HTTP ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


@pytest.fixture(scope="session", autouse=True)
def _require_stack_up(http: requests.Session) -> None:
    """Fail fast with a clear message if the stack isn't running."""
    try:
        r = http.get(f"{BASE_URL}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        pytest.exit(
            f"Serving stack is not reachable at {BASE_URL}. "
            f"Run `docker compose up -d` first.\nUnderlying error: {e}",
            returncode=2,
        )


# ── DB ────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(DATABASE_URL, future=True)
    yield engine
    engine.dispose()


@pytest.fixture
def db(db_engine) -> Iterator[Session]:
    SessionLocal = sessionmaker(bind=db_engine, autoflush=False, future=True)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ── Temp resources (auto cleanup) ─────────────────────────────────────────────

@pytest.fixture
def temp_item(http: requests.Session, db: Session) -> Iterator[int]:
    """Create one disposable item via the API; delete it (and cascading rows) after the test."""
    payload = {
        "category1": "_test_cat",
        "category2": None,
        "brand": "_test_brand",
        "price": 1.0,
    }
    r = http.post(f"{BASE_URL}/item", json=payload, timeout=10)
    r.raise_for_status()
    item_id = r.json()["item_id"]

    yield item_id

    db.execute(text("DELETE FROM interaction WHERE item_id = :iid"), {"iid": item_id})
    db.execute(text("DELETE FROM item_embedding WHERE item_id = :iid"), {"iid": item_id})
    db.execute(text("DELETE FROM item WHERE item_id = :iid"), {"iid": item_id})
    db.commit()


# A known item_id from the seeded training set. Must be < model item_embedding size,
# otherwise /recommend → IndexError. Newly-created items (via temp_item) are cold-start
# for the model and will fail inference, so they cannot be used to seed temp_user.
SEEDED_ITEM_ID = 1


@pytest.fixture
def temp_user(http: requests.Session, db: Session) -> Iterator[int]:
    """Create one disposable user with a single seeded interaction; cleanup after the test."""
    r = http.post(
        f"{BASE_URL}/user",
        json={"item_sequence": [SEEDED_ITEM_ID]},
        timeout=10,
    )
    r.raise_for_status()
    user_id = r.json()["user_id"]

    yield user_id

    db.execute(text("DELETE FROM recommendation_feedback_log WHERE user_id = :uid"), {"uid": user_id})
    db.execute(text("DELETE FROM recommendation_log WHERE user_id = :uid"), {"uid": user_id})
    db.execute(text("DELETE FROM user_representation WHERE user_id = :uid"), {"uid": user_id})
    db.execute(text("DELETE FROM interaction WHERE user_id = :uid"), {"uid": user_id})
    db.execute(text('DELETE FROM "user" WHERE user_id = :uid'), {"uid": user_id})
    db.commit()


# ── Markers ───────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "llm: tests that call the Gemini API (require GEMINI_API_KEY)")
    config.addinivalue_line("markers", "integration: tests that require the live serving stack")
