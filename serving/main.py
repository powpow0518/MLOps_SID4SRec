import os
import pickle
import torch
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse
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


_LANG_CONTENT = {
    "en": {
        "title": "MLOps SID4SRec API",
        "tags": {
            "recommendation":  "Recommendation endpoints",
            "interaction":     "Interaction recording",
            "RAG_explanation": "RAG-based recommendation explanation",
            "user_management": "User and item management",
            "system":          "System health",
        },
        "endpoints": {
            ("GET",  "/health"):      ("Health Check",            ""),
            ("GET",  "/user_list"):   ("List Users",              "Return all user IDs in the database."),
            ("GET",  "/recommend"):   ("Recommend (from DB)",     "Fetch interaction history from DB, then run inference."),
            ("POST", "/feedback"):    ("Record Feedback",         "Record a recommendation-driven interaction and check if it was a hit."),
            ("POST", "/interaction"): ("Record Interaction",      "Record an organic interaction (user found item independently)."),
            ("GET",  "/explain"):     ("Explain Recommendations", "RAG-based explanation of why these items were recommended.\n\nFlow: user_representation → similar users (cosine ≥ 0.5, top 3) → LLM → natural language explanation."),
            ("POST", "/user"):        ("Create User",             "Register a new user with their initial item sequence. Validates all item IDs exist, auto-generates user_id."),
            ("POST", "/item"):        ("Create Item",             "Create a new item. Category and brand are looked up by name and created if they don't exist."),
        },
    },
    "zh": {
        "title": "MLOps SID4SRec API",
        "tags": {
            "recommendation":  "推薦端點",
            "interaction":     "互動紀錄",
            "RAG_explanation": "RAG 推薦解釋",
            "user_management": "用戶與商品管理",
            "system":          "系統健康",
        },
        "endpoints": {
            ("GET",  "/health"):      ("健康檢查",       ""),
            ("GET",  "/user_list"):   ("列出所有用戶",   "回傳資料庫中所有 user ID。"),
            ("GET",  "/recommend"):   ("推薦（從 DB）",  "從 DB 查詢歷史互動紀錄，執行推薦推論。"),
            ("POST", "/feedback"):    ("記錄回饋",       "記錄推薦驅動的互動，並判斷是否命中推薦清單。"),
            ("POST", "/interaction"): ("記錄自然互動",   "記錄使用者自行搜尋購買的互動（非推薦來源）。"),
            ("GET",  "/explain"):     ("解釋推薦原因",   "RAG 自然語言解釋為什麼系統推薦這些商品。\n\n流程：user_representation → 相似用戶（cosine ≥ 0.5，top 3）→ LLM → 自然語言解釋。"),
            ("POST", "/user"):        ("建立新用戶",     "建立新用戶並寫入初始 item sequence。驗證所有 item ID 存在，user_id 由 DB 自動產生。"),
            ("POST", "/item"):        ("建立新商品",     "建立新商品。category 與 brand 以名稱查詢，不存在則自動建立。"),
        },
    },
}

app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    openapi_url=None,
)


def _build_openapi(lang: str = "en") -> dict:
    content = _LANG_CONTENT[lang]
    schema = get_openapi(
        title=content["title"],
        version="1.0.0",
        routes=app.routes,
    )
    schema["tags"] = [
        {"name": name, "description": desc}
        for name, desc in content["tags"].items()
    ]
    for path, methods in schema.get("paths", {}).items():
        for method, operation in methods.items():
            key = (method.upper(), path)
            if key in content["endpoints"]:
                summary, description = content["endpoints"][key]
                operation["summary"] = summary
                operation.pop("description", None)
                if description:
                    operation["description"] = description
    return schema


@app.get("/openapi.json", include_in_schema=False)
def openapi_en():
    return JSONResponse(_build_openapi("en"))


@app.get("/openapi-zh.json", include_in_schema=False)
def openapi_zh():
    return JSONResponse(_build_openapi("zh"))


@app.get("/docs", include_in_schema=False)
def custom_docs():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>MLOps SID4SRec API</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
        #lang-btn {
            position: fixed;
            top: 12px;
            right: 20px;
            z-index: 9999;
            background: #49cc90;
            color: white;
            border: none;
            padding: 8px 18px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        #lang-btn:hover { background: #3bab74; }
    </style>
</head>
<body>
<button id="lang-btn" onclick="switchLang()">切換至中文</button>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
<script>
    let currentLang = 'en';

    function initUI(url) {
        SwaggerUIBundle({
            url: url,
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
            layout: 'StandaloneLayout',
            deepLinking: true,
        });
    }

    function switchLang() {
        currentLang = currentLang === 'en' ? 'zh' : 'en';
        const url = currentLang === 'en' ? '/openapi.json' : '/openapi-zh.json';
        document.getElementById('lang-btn').textContent =
            currentLang === 'en' ? '切換至中文' : 'Switch to English';
        initUI(url);
    }

    window.onload = () => initUI('/openapi.json');
</script>
</body>
</html>
"""
    return HTMLResponse(html)


# ── Request schemas ───────────────────────────────────────────────────────────
class CreateUserRequest(BaseModel):
    item_sequence: List[int]


class CreateItemRequest(BaseModel):
    category1: str
    category2: Optional[str] = None
    brand: str
    price: float


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
def _run_inference(item_sequence: List[int], top_k: int = 20):
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
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.get("/user_list", tags=["recommendation"])
def user_list(db: Session = Depends(get_db)):
    rows = db.execute(text('SELECT user_id FROM "user" ORDER BY user_id')).fetchall()
    return {"users": [r[0] for r in rows]}


@app.get("/recommend", tags=["recommendation"])
def recommend_from_db(user_id: int, db: Session = Depends(get_db)):
    """Fetch interaction history from DB, then run inference."""
    rows = db.execute(
        text("SELECT item_id FROM interaction WHERE user_id = :uid ORDER BY timestamp ASC"),
        {"uid": user_id},
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No history found for user {user_id}")

    item_sequence = [r[0] for r in rows]
    recommended, user_repr = _run_inference(item_sequence)
    _save_recommendation_log(db, user_id, recommended)
    _upsert_user_representation(db, user_id, user_repr)
    db.commit()
    return {"user_id": user_id, "recommendations": recommended}


@app.post("/user", tags=["user_management"], status_code=201)
def create_user(req: CreateUserRequest, db: Session = Depends(get_db)):
    """Register a new user with their initial item sequence.

    Validates all item_ids exist, auto-generates user_id, writes interactions to DB.
    """
    # Validate all items exist
    for item_id in req.item_sequence:
        exists = db.execute(
            text("SELECT 1 FROM item WHERE item_id = :iid"),
            {"iid": item_id},
        ).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail=f"Item {item_id} does not exist")

    # Auto-generate user_id — nextval is atomic, safe for concurrent requests
    row = db.execute(text("SELECT nextval('user_id_seq')")).fetchone()
    new_user_id = row[0]

    db.execute(text('INSERT INTO "user" (user_id) VALUES (:uid)'), {"uid": new_user_id})

    now = datetime.now(timezone.utc)
    for item_id in req.item_sequence:
        db.execute(
            text("INSERT INTO interaction (user_id, item_id, timestamp) VALUES (:uid, :iid, :ts)"),
            {"uid": new_user_id, "iid": item_id, "ts": now},
        )

    db.commit()
    return {"user_id": new_user_id}


@app.post("/item", tags=["user_management"], status_code=201)
def create_item(req: CreateItemRequest, db: Session = Depends(get_db)):
    """Create a new item with category, brand, and price.

    Category and brand are looked up by name; created if they don't exist.
    """
    # Upsert category1
    row = db.execute(
        text("SELECT category_id FROM category WHERE category = :name"),
        {"name": req.category1},
    ).fetchone()
    if row:
        category_id1 = row[0]
    else:
        r = db.execute(
            text("INSERT INTO category (category_id, category) VALUES (nextval('category_id_seq'), :name) RETURNING category_id"),
            {"name": req.category1},
        ).fetchone()
        category_id1 = r[0]

    # Upsert category2 (optional)
    category_id2 = None
    if req.category2:
        row = db.execute(
            text("SELECT category_id FROM category WHERE category = :name"),
            {"name": req.category2},
        ).fetchone()
        if row:
            category_id2 = row[0]
        else:
            r = db.execute(
                text("INSERT INTO category (category_id, category) VALUES (nextval('category_id_seq'), :name) RETURNING category_id"),
                {"name": req.category2},
            ).fetchone()
            category_id2 = r[0]

    # Upsert brand
    row = db.execute(
        text("SELECT brand_id FROM brand WHERE brand_name = :name"),
        {"name": req.brand},
    ).fetchone()
    if row:
        brand_id = row[0]
    else:
        r = db.execute(
            text("INSERT INTO brand (brand_id, brand_name) VALUES (nextval('brand_id_seq'), :name) RETURNING brand_id"),
            {"name": req.brand},
        ).fetchone()
        brand_id = r[0]

    # Insert item
    r = db.execute(
        text("""
            INSERT INTO item (item_id, category_id1, category_id2, brand_id, price)
            VALUES (nextval('item_id_seq'), :c1, :c2, :bid, :price)
            RETURNING item_id
        """),
        {"c1": category_id1, "c2": category_id2, "bid": brand_id, "price": req.price},
    ).fetchone()

    db.commit()
    return {"item_id": r[0]}


@app.post("/feedback", tags=["interaction"])
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


@app.get("/explain", tags=["RAG_explanation"])
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


@app.post("/interaction", tags=["interaction"])
def interaction(req: InteractionRequest, db: Session = Depends(get_db)):
    """Record an organic interaction (user found item independently)."""
    now = datetime.now(timezone.utc)
    db.execute(
        text("INSERT INTO interaction (user_id, item_id, timestamp) VALUES (:uid, :iid, :ts)"),
        {"uid": req.user_id, "iid": req.item_id, "ts": now},
    )
    db.commit()
    return {"status": "recorded"}
