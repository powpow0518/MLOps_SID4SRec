"""
Integration tests for the serving API (FastAPI behind Nginx).

Run with the stack up:
    docker compose up -d
    pytest tests/test_api.py

Skip LLM-dependent tests:
    pytest tests/test_api.py -m "not llm"

Notes on validation gaps marked with `xfail`:
    /feedback and /interaction don't validate that user_id / item_id exist —
    bad inputs currently surface as 500 from a DB FK violation. The xfail tests
    pin this gap so it shows up in CI; once the endpoints validate properly,
    flip them to expected pass and they'll fail loudly to remind us to remove the marker.
"""

import os

import pytest
import requests

pytestmark = pytest.mark.integration


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_ok(self, http: requests.Session, base_url: str):
        r = http.get(f"{base_url}/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ── GET /recommend ────────────────────────────────────────────────────────────

class TestRecommend:
    def test_valid_user_returns_20_items(self, http, base_url, temp_user):
        # temp_user has one interaction → enough for inference
        r = http.get(f"{base_url}/recommend", params={"user_id": temp_user})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == temp_user
        assert isinstance(body["recommendations"], list)
        assert len(body["recommendations"]) == 20
        assert all(isinstance(i, int) for i in body["recommendations"])

    def test_user_with_no_history_returns_404(self, http, base_url):
        # Astronomically unlikely to exist
        r = http.get(f"{base_url}/recommend", params={"user_id": 99_999_999})
        assert r.status_code == 404

    @pytest.mark.parametrize("bad_user_id", ["abc", "", "1; DROP TABLE user;--"])
    def test_non_int_user_id_rejected(self, http, base_url, bad_user_id):
        r = http.get(f"{base_url}/recommend", params={"user_id": bad_user_id})
        assert r.status_code == 422

    def test_missing_user_id_rejected(self, http, base_url):
        r = http.get(f"{base_url}/recommend")
        assert r.status_code == 422


# ── POST /user ────────────────────────────────────────────────────────────────

class TestCreateUser:
    def test_valid_payload_creates_user(self, http, base_url, temp_item):
        r = http.post(f"{base_url}/user", json={"item_sequence": [temp_item]})
        assert r.status_code == 201
        body = r.json()
        assert isinstance(body["user_id"], int)
        assert body["user_id"] > 0

        # Cleanup (no temp_user fixture here since we're testing the create itself)
        # Done via DB in the fixture-based tests; here we accept the leak — see test_create_user_cleanup below

    def test_nonexistent_item_rejected(self, http, base_url):
        r = http.post(f"{base_url}/user", json={"item_sequence": [99_999_999]})
        assert r.status_code == 404

    @pytest.mark.parametrize("bad_payload", [
        {},                                  # missing field
        {"item_sequence": None},             # null
        {"item_sequence": "abc"},            # not a list
        {"item_sequence": [1, "abc", 3]},    # mixed types
    ])
    def test_invalid_payload_rejected(self, http, base_url, bad_payload):
        r = http.post(f"{base_url}/user", json=bad_payload)
        assert r.status_code == 422


# ── POST /item ────────────────────────────────────────────────────────────────

class TestCreateItem:
    def test_valid_payload_creates_item(self, http, base_url, temp_item):
        # temp_item fixture already exercised the happy path
        assert isinstance(temp_item, int)
        assert temp_item > 0

    def test_existing_category_is_reused(self, http, base_url, db, temp_item):
        """Two items sharing a category name should map to the same category_id."""
        r = http.post(f"{base_url}/item", json={
            "category1": "_test_cat",      # same as temp_item
            "category2": None,
            "brand": "_test_brand",
            "price": 2.0,
        })
        assert r.status_code == 201
        new_id = r.json()["item_id"]

        from sqlalchemy import text
        rows = db.execute(
            text("SELECT category_id1 FROM item WHERE item_id IN (:a, :b)"),
            {"a": temp_item, "b": new_id},
        ).fetchall()
        assert len({r[0] for r in rows}) == 1, "category should not be duplicated"

        db.execute(text("DELETE FROM item WHERE item_id = :iid"), {"iid": new_id})
        db.commit()

    @pytest.mark.parametrize("bad_payload", [
        {},                                                                       # all missing
        {"category2": None, "brand": "b", "price": 1.0},                          # category1 missing
        {"category1": "c", "category2": None, "price": 1.0},                      # brand missing
        {"category1": "c", "category2": None, "brand": "b"},                      # price missing
        {"category1": 123, "category2": None, "brand": "b", "price": 1.0},        # category1 type
        {"category1": "c", "category2": None, "brand": 123, "price": 1.0},        # brand type
        {"category1": "c", "category2": None, "brand": "b", "price": "abc"},      # price type
    ])
    def test_invalid_payload_rejected(self, http, base_url, bad_payload):
        r = http.post(f"{base_url}/item", json=bad_payload)
        assert r.status_code == 422

    @pytest.mark.xfail(reason="endpoint does not validate price >= 0; gap pinned for follow-up")
    def test_negative_price_rejected(self, http, base_url):
        r = http.post(f"{base_url}/item", json={
            "category1": "_t", "category2": None, "brand": "_t", "price": -1.0,
        })
        assert r.status_code == 422


# ── POST /feedback ────────────────────────────────────────────────────────────

class TestFeedback:
    def test_valid_feedback_with_recommendation_marks_hit(self, http, base_url, temp_user, db):
        # First call /recommend so a recommendation_log row exists
        r = http.get(f"{base_url}/recommend", params={"user_id": temp_user})
        assert r.status_code == 200
        recommended = r.json()["recommendations"]

        # Feedback on a recommended item → hit = True
        r = http.post(f"{base_url}/feedback", json={"user_id": temp_user, "item_id": recommended[0]})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "recorded"
        assert body["hit"] is True

    def test_valid_feedback_not_in_recommendations(self, http, base_url, temp_user, temp_item):
        http.get(f"{base_url}/recommend", params={"user_id": temp_user})
        # Use a known item not in the recommendation list (temp_item is freshly created → not in any rec)
        r = http.post(f"{base_url}/feedback", json={"user_id": temp_user, "item_id": temp_item})
        assert r.status_code == 200
        assert r.json()["hit"] is False

    @pytest.mark.parametrize("bad_payload", [
        {},
        {"user_id": 1},
        {"item_id": 1},
        {"user_id": None, "item_id": 1},
        {"user_id": "abc", "item_id": 1},
        {"user_id": 1, "item_id": "abc"},
    ])
    def test_invalid_payload_rejected(self, http, base_url, bad_payload):
        r = http.post(f"{base_url}/feedback", json=bad_payload)
        assert r.status_code == 422

    @pytest.mark.xfail(reason="endpoint does not validate user_id/item_id existence; FK violation surfaces as 500")
    def test_nonexistent_user_rejected(self, http, base_url):
        r = http.post(f"{base_url}/feedback", json={"user_id": 99_999_999, "item_id": 1})
        assert r.status_code in (404, 422)


# ── POST /interaction ─────────────────────────────────────────────────────────

class TestInteraction:
    def test_valid_interaction_recorded(self, http, base_url, temp_user, temp_item):
        r = http.post(f"{base_url}/interaction", json={"user_id": temp_user, "item_id": temp_item})
        assert r.status_code == 200
        assert r.json()["status"] == "recorded"

    @pytest.mark.parametrize("bad_payload", [
        {},
        {"user_id": 1},
        {"item_id": 1},
        {"user_id": None, "item_id": 1},
        {"user_id": "abc", "item_id": 1},
        {"user_id": 1, "item_id": "abc"},
    ])
    def test_invalid_payload_rejected(self, http, base_url, bad_payload):
        r = http.post(f"{base_url}/interaction", json=bad_payload)
        assert r.status_code == 422

    @pytest.mark.xfail(reason="endpoint does not validate user_id/item_id existence; FK violation surfaces as 500")
    def test_nonexistent_user_rejected(self, http, base_url):
        r = http.post(f"{base_url}/interaction", json={"user_id": 99_999_999, "item_id": 1})
        assert r.status_code in (404, 422)


# ── GET /explain ──────────────────────────────────────────────────────────────

class TestExplain:
    def test_user_without_representation_returns_404(self, http, base_url, temp_user):
        # temp_user exists but has not called /recommend → no representation
        r = http.get(f"{base_url}/explain", params={"user_id": temp_user})
        assert r.status_code == 404

    def test_missing_user_id_rejected(self, http, base_url):
        r = http.get(f"{base_url}/explain")
        assert r.status_code == 422

    @pytest.mark.parametrize("bad_user_id", ["abc", "1; DROP TABLE user;--"])
    def test_non_int_user_id_rejected(self, http, base_url, bad_user_id):
        r = http.get(f"{base_url}/explain", params={"user_id": bad_user_id})
        assert r.status_code == 422

    def test_invalid_lang_rejected(self, http, base_url, temp_user):
        r = http.get(f"{base_url}/explain", params={"user_id": temp_user, "lang": "jp"})
        assert r.status_code == 422

    @pytest.mark.llm
    @pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
    @pytest.mark.parametrize("lang", ["zh", "en"])
    def test_explain_returns_text_after_recommend(self, http, base_url, temp_user, lang):
        # Must call /recommend first to populate user_representation
        r = http.get(f"{base_url}/recommend", params={"user_id": temp_user})
        assert r.status_code == 200

        r = http.get(f"{base_url}/explain", params={"user_id": temp_user, "lang": lang})
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == temp_user
        assert "summary" in body
        assert isinstance(body["summary"], str)
        assert len(body["summary"]) > 0
        assert body.get("source") in ("llm", "fallback")
