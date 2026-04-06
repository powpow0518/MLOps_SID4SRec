import os
import pickle
import torch
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Literal

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from training.sid4srec import SID4SRec
from rag.explain import explain_user

# ── Global model state ───────────────────────────────────────────────────────
_model: SID4SRec = None
_device: torch.device = None
_args = None
_model_version: str = None   # 目前 active 的 model version，啟動時從 DB 快取

DATABASE_URL = os.environ["DATABASE_URL"]
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best_model.pt")
# model_args.pkl is saved alongside model weights during training (see trainer.py)
MODEL_ARGS_PATH = os.getenv("MODEL_ARGS_PATH", "/models/model_args.pkl")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _create_tables():
    """Create tables that are not part of the core schema but needed for serving."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS recommendation_log (
                id                SERIAL    PRIMARY KEY,
                user_id           INTEGER   NOT NULL,
                recommended_items INTEGER[] NOT NULL,
                created_at        TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        conn.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _args, _model_version

    _create_tables()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(MODEL_ARGS_PATH, "rb") as f:
        _args = pickle.load(f)

    # Move lookup tensors to device
    _args.category_lookup = _args.category_lookup.to(_device)
    _args.brand_lookup = _args.brand_lookup.to(_device)

    _model = SID4SRec(_device, _args)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    _model.to(_device)
    _model.eval()

    # Cache active model version from DB (used for user_representation UPSERT)
    with SessionLocal() as db:
        row = db.execute(
            text("SELECT model_version FROM model_version WHERE is_active = TRUE LIMIT 1")
        ).fetchone()
        _model_version = row[0] if row else None
        if _model_version is None:
            print("Warning: no active model version in DB — user_representation UPSERT will be skipped")

    yield


app = FastAPI(lifespan=lifespan)


# ── Request schemas ───────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    user_id: int
    item_sequence: List[int]


class FeedbackRequest(BaseModel):
    user_id: int
    item_id: int


class InteractionRequest(BaseModel):
    user_id: int
    item_id: int


# ── DB dependency ─────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Inference helper ──────────────────────────────────────────────────────────
def _run_inference(item_sequence: List[int], top_k: int = 10):
    """Run inference and return (top_k item ids, user representation as list).

    The user representation is the transformer's last-position output (dim=192),
    used for RAG similarity search via user_representation table.
    """
    max_len = _args.max_seq_length
    seq = item_sequence[-max_len:]
    padded = [0] * (max_len - len(seq)) + seq
    input_tensor = torch.tensor([padded], dtype=torch.long).to(_device)

    with torch.no_grad():
        user_repr = _model.get_user_representation(input_tensor)   # [1, 192]
        scores = torch.matmul(user_repr, _model.items_emb().transpose(0, 1))[0].cpu()

    # Exclude items already seen
    for item_id in item_sequence:
        if 0 < item_id < scores.shape[0]:
            scores[item_id] = -1e9

    top_items = torch.argsort(scores, descending=True)[:top_k].tolist()
    return top_items, user_repr[0].cpu().tolist()


def _upsert_user_representation(db: Session, user_id: int, representation: List[float]):
    if _model_version is None:
        return
    vector_str = "[" + ",".join(str(v) for v in representation) + "]"
    db.execute(
        text("""
            INSERT INTO user_representation (user_id, model_version, representation)
            VALUES (:uid, :mv, CAST(:repr AS vector))
            ON CONFLICT (user_id, model_version) DO UPDATE
                SET representation = EXCLUDED.representation,
                    created_at     = NOW()
        """),
        {"uid": user_id, "mv": _model_version, "repr": vector_str},
    )


def _save_recommendation_log(db: Session, user_id: int, items: List[int]):
    # Convert Python list to PostgreSQL array literal e.g. "{1,2,3}"
    array_literal = "{" + ",".join(str(i) for i in items) + "}"
    db.execute(
        text("INSERT INTO recommendation_log (user_id, recommended_items) VALUES (:uid, :items)"),
        {"uid": user_id, "items": array_literal},
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/user_list")
def user_list(db: Session = Depends(get_db)):
    rows = db.execute(text('SELECT user_id FROM "user" ORDER BY user_id')).fetchall()
    return {"users": [r[0] for r in rows]}


@app.get("/recommend")
def recommend_from_db(user_id: int, top_k: int = 10, db: Session = Depends(get_db)):
    """Fetch interaction history from DB, then run inference."""
    rows = db.execute(
        text("SELECT item_id FROM interaction WHERE user_id = :uid ORDER BY timestamp ASC"),
        {"uid": user_id},
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No history found for user {user_id}")

    item_sequence = [r[0] for r in rows]
    recommended, user_repr = _run_inference(item_sequence, top_k=top_k)
    _save_recommendation_log(db, user_id, recommended)
    _upsert_user_representation(db, user_id, user_repr)
    db.commit()
    return {"user_id": user_id, "recommendations": recommended}


@app.post("/recommend")
def recommend_from_sequence(req: RecommendRequest, top_k: int = 10, db: Session = Depends(get_db)):
    """Accept item sequence directly, run inference without querying DB."""
    recommended, user_repr = _run_inference(req.item_sequence, top_k=top_k)
    _save_recommendation_log(db, req.user_id, recommended)
    _upsert_user_representation(db, req.user_id, user_repr)
    db.commit()
    return {"user_id": req.user_id, "recommendations": recommended}


@app.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Record a recommendation-driven interaction and check if it was a hit."""
    now = datetime.now(timezone.utc)

    db.execute(
        text("INSERT INTO interaction (user_id, item_id, timestamp) VALUES (:uid, :iid, :ts)"),
        {"uid": req.user_id, "iid": req.item_id, "ts": now},
    )

    # Check if the item appeared in the most recent recommendation for this user
    row = db.execute(
        text("""
            SELECT :item_id = ANY(recommended_items)
            FROM recommendation_log
            WHERE user_id = :uid
            ORDER BY created_at DESC
            LIMIT 1
        """),
        {"uid": req.user_id, "item_id": req.item_id},
    ).fetchone()

    hit = bool(row[0]) if row else False

    db.execute(
        text("""
            INSERT INTO recommendation_feedback_log (user_id, item_id, timestamp, hit)
            VALUES (:uid, :iid, :ts, :hit)
        """),
        {"uid": req.user_id, "iid": req.item_id, "ts": now, "hit": hit},
    )

    db.commit()
    return {"status": "recorded", "hit": hit}


@app.get("/explain")
def explain(
    user_id: int,
    lang: Literal["zh", "en"] = "zh",
    db: Session = Depends(get_db),
):
    """RAG 解釋為什麼系統推薦這些 item 給此 user。

    流程：
    1. 確認 user_representation 存在（不存在 → 404，請先呼叫 /recommend）
    2. 取最新一筆推薦 + HNSW 找相似 user（cosine ≥ 0.5, top 3）
    3. 組裝 context（每個 user 最近 10 個 item + top 3 cats/brands）
    4. 呼叫 Gemini API → 自然語言解釋

    Response: {user_id, explanation, source: "llm" | "fallback"}
    """
    result = explain_user(db, user_id, lang)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} has no representation. Call POST /recommend first.",
        )
    return {"user_id": user_id, **result}


@app.post("/interaction")
def interaction(req: InteractionRequest, db: Session = Depends(get_db)):
    """Record an organic interaction (user found item independently)."""
    now = datetime.now(timezone.utc)
    db.execute(
        text("INSERT INTO interaction (user_id, item_id, timestamp) VALUES (:uid, :iid, :ts)"),
        {"uid": req.user_id, "iid": req.item_id, "ts": now},
    )
    db.commit()
    return {"status": "recorded"}
