"""
RAG context fetching: 從 DB 取得 RAG 所需的所有資料，組成 dataclass。

每個 user 的 context = 最近 10 個 item（含屬性）+ top 3 categories + top 3 brands。
RagContext = 目標 user + 相似 users + 推薦 items。
"""

from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

SIMILAR_USER_TOP_K = 3
SIMILAR_USER_MIN_SIM = 0.5
RECENT_ITEMS_LIMIT = 10
TOP_N_CATS_BRANDS = 3


@dataclass
class ItemAttrs:
    item_id: int
    category: str
    brand: str
    price: float


@dataclass
class UserContext:
    user_id: int
    recent_items: list[ItemAttrs] = field(default_factory=list)
    top_categories: list[tuple[str, int]] = field(default_factory=list)
    top_brands: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class RagContext:
    target_user: UserContext
    similar_users: list[UserContext]
    recommended_items: list[ItemAttrs]


# ── DB queries ───────────────────────────────────────────────────────────────


def has_user_representation(db: Session, user_id: int) -> bool:
    row = db.execute(
        text("SELECT 1 FROM user_representation WHERE user_id = :uid LIMIT 1"),
        {"uid": user_id},
    ).fetchone()
    return row is not None


def get_latest_recommendations(db: Session, user_id: int) -> list[int]:
    row = db.execute(
        text(
            "SELECT recommended_items FROM recommendation_log "
            "WHERE user_id = :uid ORDER BY created_at DESC LIMIT 1"
        ),
        {"uid": user_id},
    ).fetchone()
    return list(row[0]) if row else []


def find_similar_users(
    db: Session,
    user_id: int,
    top_k: int = SIMILAR_USER_TOP_K,
    min_sim: float = SIMILAR_USER_MIN_SIM,
) -> list[int]:
    """HNSW 向量搜尋：cosine similarity ≥ min_sim 的 top_k 個 user（不含自己）。

    pgvector 用 <=> 算 cosine distance（= 1 - cosine_similarity）。
    """
    rows = db.execute(
        text(
            """
            WITH target AS (
                SELECT representation FROM user_representation
                WHERE user_id = :uid
                ORDER BY created_at DESC LIMIT 1
            )
            SELECT u.user_id, 1 - (u.representation <=> t.representation) AS sim
            FROM user_representation u, target t
            WHERE u.user_id <> :uid
            ORDER BY u.representation <=> t.representation
            LIMIT :k
            """
        ),
        {"uid": user_id, "k": top_k},
    ).fetchall()
    return [r[0] for r in rows if r[1] >= min_sim]


def _row_to_item(row) -> ItemAttrs:
    return ItemAttrs(
        item_id=row[0],
        category=row[1] or "unknown",
        brand=row[2] or "unknown",
        price=float(row[3]) if row[3] is not None else 0.0,
    )


def get_item_attrs(db: Session, item_ids: list[int]) -> list[ItemAttrs]:
    """取得指定 item_ids 的屬性，保留輸入順序。"""
    if not item_ids:
        return []
    rows = db.execute(
        text(
            """
            SELECT i.item_id, c.category, b.brand_name, i.price
            FROM item i
            LEFT JOIN category c ON i.category_id1 = c.category_id
            LEFT JOIN brand b ON i.brand_id = b.brand_id
            WHERE i.item_id = ANY(:ids)
            """
        ),
        {"ids": item_ids},
    ).fetchall()
    by_id = {r[0]: _row_to_item(r) for r in rows}
    return [by_id[iid] for iid in item_ids if iid in by_id]


def get_user_context(db: Session, user_id: int) -> UserContext:
    """組裝單一 user 的 context：最近 10 個 item + top 3 cats/brands。"""
    rows = db.execute(
        text(
            """
            SELECT i.item_id, c.category, b.brand_name, i.price
            FROM interaction inter
            JOIN item i ON inter.item_id = i.item_id
            LEFT JOIN category c ON i.category_id1 = c.category_id
            LEFT JOIN brand b ON i.brand_id = b.brand_id
            WHERE inter.user_id = :uid
            ORDER BY inter.timestamp ASC
            """
        ),
        {"uid": user_id},
    ).fetchall()

    if not rows:
        return UserContext(user_id=user_id)

    recent_rows = rows[-RECENT_ITEMS_LIMIT:]
    recent_items = [_row_to_item(r) for r in recent_rows]

    cat_counter = Counter(r[1] for r in rows if r[1])
    brand_counter = Counter(r[2] for r in rows if r[2])

    return UserContext(
        user_id=user_id,
        recent_items=recent_items,
        top_categories=cat_counter.most_common(TOP_N_CATS_BRANDS),
        top_brands=brand_counter.most_common(TOP_N_CATS_BRANDS),
    )


def build_rag_context(db: Session, user_id: int) -> RagContext | None:
    """主入口：組完整的 RagContext。若 user_representation 不存在則回 None。"""
    if not has_user_representation(db, user_id):
        return None

    rec_item_ids = get_latest_recommendations(db, user_id)
    similar_user_ids = find_similar_users(db, user_id)

    return RagContext(
        target_user=get_user_context(db, user_id),
        similar_users=[get_user_context(db, uid) for uid in similar_user_ids],
        recommended_items=get_item_attrs(db, rec_item_ids),
    )
