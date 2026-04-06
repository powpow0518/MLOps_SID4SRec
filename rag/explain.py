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

SYSTEM_PROMPT = {
    "zh": (
        "你是推薦系統的解釋助理。根據以下用戶資料，用繁體中文寫『一段』自然、簡潔的推薦解釋，"
        "說明系統為什麼推薦這些商品給此用戶，必須涵蓋全部推薦商品。"
        "請從『品牌偏好』、『類別偏好』、『相似用戶行為』三個角度找出共同點。\n\n"
        "【極重要 — 輸出格式規則，違反則視為失敗】\n"
        "1. 只輸出『最終解釋文字』本身，完全不要顯示任何思考過程、草稿、自我修正、檢查、或標記。\n"
        "2. 禁止出現以下內容：'Draft', 'Final', 'Self-Correction', 'Refining', 'Internal Monologue', "
        "'Wait', 'Let me', 'Re-evaluating', '*', '**', 條列符號、標題、前綴。\n"
        "3. 禁止使用 markdown 格式（不要用 *、-、#、`）。\n"
        "4. 直接從第一個字開始就是給用戶看的解釋本身，整段不超過 200 字。\n"
        "5. 整個回應只能有『一段』純文字，不能有多段、不能有換行分隔的多個版本。"
    ),
    "en": (
        "You are an explanation assistant for a recommendation system. Based on the following user data, "
        "write ONE concise, natural-language paragraph in English explaining why these items are recommended "
        "to this user, covering all recommended items. Find commonalities from the angles of brand preference, "
        "category preference, and similar user behavior.\n\n"
        "[CRITICAL — Output format rules, violation = failure]\n"
        "1. Output ONLY the final explanation text. Do NOT show any thinking, drafts, self-corrections, "
        "checks, or annotations.\n"
        "2. Forbidden: 'Draft', 'Final', 'Self-Correction', 'Refining', 'Internal Monologue', 'Wait', "
        "'Let me', 'Re-evaluating', '*', '**', bullet points, titles, prefixes.\n"
        "3. No markdown (no *, -, #, `).\n"
        "4. Start directly with the explanation. Max 200 words.\n"
        "5. Only ONE paragraph of plain text. No multiple versions, no separators."
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


def build_prompt(ctx: RagContext, lang: str) -> str:
    parts = [SYSTEM_PROMPT[lang], ""]

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


# ── Main entry ───────────────────────────────────────────────────────────────


def explain_user(db: Session, user_id: int, lang: str = "zh") -> Optional[dict]:
    """
    Returns:
        None — user_representation 不存在（caller 應回 404）
        {"explanation": str, "source": "llm" | "fallback"} — 成功或 LLM 失敗 fallback
    """
    ctx = build_rag_context(db, user_id)
    if ctx is None:
        return None

    prompt = build_prompt(ctx, lang)

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return {"explanation": response.text.strip(), "source": "llm"}
    except Exception:
        logger.exception("Gemini API call failed for user_id=%s", user_id)
        return {"explanation": FALLBACK_TEXT[lang], "source": "fallback"}
