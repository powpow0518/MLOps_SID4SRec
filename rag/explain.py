"""
RAG explanation: 把 RagContext 組成 prompt，呼叫 Gemini API，回傳自然語言解釋。

回傳格式：{"explanation": str, "source": "llm" | "fallback"}
- llm 失敗（API error / quota / network）時 fallback 並 log，但仍回 200 給 client。
"""

import logging
import os
from typing import Optional

from sqlalchemy.orm import Session

from rag.context import RagContext, UserContext, build_rag_context

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemma-4-31b-it")

_gemini_model = None


def _get_gemini_model():
    """Lazy init Gemini client（避免 import 時就要 API key）。"""
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set")
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    return _gemini_model


# ── Prompt templates ─────────────────────────────────────────────────────────

STRUCTURED_SYSTEM_PROMPT = {
    "zh": (
        "你是推薦系統的解釋助理。根據以下用戶資料，針對每一個推薦商品，用繁體中文寫一句話解釋為什麼推薦給此用戶。"
        "可以從品牌偏好、類別偏好、相似用戶行為三個角度切入。\n\n"
        "【輸出格式規則】\n"
        "1. 每個商品一行，格式為：item_id=X（brand: B, category: C）：一句話原因\n"
        "2. 不要顯示思考過程、草稿或任何額外說明。\n"
        "3. 禁止 markdown（不要用 *、-、#、`）。\n"
        "4. 直接從第一個商品開始輸出。"
    ),
    "en": (
        "You are an explanation assistant for a recommendation system. Based on the following user data, "
        "write ONE sentence per recommended item explaining why it was recommended to this user. "
        "Draw from brand preference, category preference, and similar user behavior.\n\n"
        "[Output format rules]\n"
        "1. One line per item: item_id=X (brand: B, category: C): one-sentence reason\n"
        "2. Do NOT show thinking, drafts, or extra commentary.\n"
        "3. No markdown (no *, -, #, `).\n"
        "4. Start directly from the first item."
    ),
}

SUMMARY_SYSTEM_PROMPT = {
    "zh": (
        "你是推薦系統的摘要助理。根據以下逐條推薦解釋，用繁體中文寫『一段』重點摘要，"
        "整合出此用戶的品牌偏好、類別偏好與相似用戶行為的核心規律，不超過 150 字。\n\n"
        "【輸出格式規則】\n"
        "1. 只輸出摘要本身，不要重複列出商品清單。\n"
        "2. 不要顯示思考過程或草稿。\n"
        "3. 禁止 markdown。\n"
        "4. 直接從第一個字開始。"
    ),
    "en": (
        "You are a summarization assistant for a recommendation system. Based on the structured explanation below, "
        "write ONE concise paragraph (max 150 words) summarizing the key patterns: brand preference, "
        "category preference, and similar user behavior insights.\n\n"
        "[Output format rules]\n"
        "1. Output only the summary — do not repeat the item list.\n"
        "2. No thinking or drafts.\n"
        "3. No markdown.\n"
        "4. Start directly."
    ),
}

FALLBACK_TEXT = {
    "zh": "推薦解釋暫時無法生成，請稍後再試。",
    "en": "Recommendation explanation is temporarily unavailable. Please try again later.",
}


def _format_items(items) -> str:
    if not items:
        return "  (none)"
    return "\n".join(
        f"  - item_id={i.item_id}, category={i.category}, brand={i.brand}, price={i.price:.2f}"
        for i in items
    )


def _format_top(top_list) -> str:
    if not top_list:
        return "(none)"
    return ", ".join(f"{name} (x{count})" for name, count in top_list)


def _format_user_block(label: str, ctx: UserContext) -> str:
    return (
        f"## {label} (user_id={ctx.user_id})\n"
        f"Recent 10 interactions:\n{_format_items(ctx.recent_items)}\n"
        f"Top 3 categories (whole history): {_format_top(ctx.top_categories)}\n"
        f"Top 3 brands (whole history): {_format_top(ctx.top_brands)}"
    )


def build_structured_prompt(ctx: RagContext, lang: str) -> str:
    parts = [STRUCTURED_SYSTEM_PROMPT[lang], ""]

    parts.append(_format_user_block("Target user", ctx.target_user))
    parts.append("")

    if ctx.similar_users:
        for i, sim in enumerate(ctx.similar_users, 1):
            parts.append(_format_user_block(f"Similar user #{i}", sim))
            parts.append("")
    else:
        parts.append("## Similar users\n  (no similar users found above similarity threshold)")
        parts.append("")

    parts.append("## Recommended items for the target user")
    parts.append(_format_items(ctx.recommended_items))

    return "\n".join(parts)


def build_summary_prompt(structured_output: str, lang: str) -> str:
    return (
        f"{SUMMARY_SYSTEM_PROMPT[lang]}\n\n"
        f"## Structured explanation\n{structured_output}"
    )


# ── Main entry ───────────────────────────────────────────────────────────────


def explain_user(db: Session, user_id: int, lang: str = "zh") -> Optional[dict]:
    """
    Returns:
        None — user_representation 不存在（caller 應回 404）
        {"structured": str, "summary": str, "source": "llm" | "fallback", ...} — 成功或 LLM 失敗 fallback
    """
    ctx = build_rag_context(db, user_id)
    if ctx is None:
        return None

    recommended = [
        {"item_id": i.item_id, "category": i.category, "brand": i.brand, "price": i.price}
        for i in ctx.recommended_items
    ]

    user_context = {
        "recent_items": [
            {"item_id": i.item_id, "category": i.category, "brand": i.brand, "price": i.price}
            for i in ctx.target_user.recent_items
        ],
        "top_categories": [{"category": name, "count": cnt} for name, cnt in ctx.target_user.top_categories],
        "top_brands":     [{"brand": name, "count": cnt} for name, cnt in ctx.target_user.top_brands],
    }

    try:
        gemini = _get_gemini_model()

        structured_prompt = build_structured_prompt(ctx, lang)
        structured = gemini.generate_content(structured_prompt).text.strip()

        summary_prompt = build_summary_prompt(structured, lang)
        summary = gemini.generate_content(summary_prompt).text.strip()

        return {
            "summary": summary,
            "source": "llm",
            "recommended_items": recommended,
            "user_context": user_context,
        }
    except Exception:
        logger.exception("Gemini API call failed for user_id=%s", user_id)
        return {
            "summary": FALLBACK_TEXT[lang],
            "source": "fallback",
            "recommended_items": recommended,
            "user_context": user_context,
        }
