"""
Tests for rag/explain.py — pure formatting functions, no DB or LLM required.
"""

import pytest
from rag.explain import (
    FALLBACK_TEXT,
    _format_items,
    _format_top,
    build_summary_prompt,
    build_structured_prompt,
)
from rag.context import ItemAttrs, RagContext, UserContext


# ── _format_items ─────────────────────────────────────────────────────────────

def test_format_items_empty():
    assert _format_items([]) == "  (none)"


def test_format_items_single():
    items = [ItemAttrs(item_id=1, category="Lotion", brand="CeraVe", price=12.99)]
    result = _format_items(items)
    assert "item_id=1" in result
    assert "Lotion" in result
    assert "CeraVe" in result
    assert "12.99" in result


def test_format_items_multiple_lines():
    items = [
        ItemAttrs(item_id=1, category="Lotion", brand="CeraVe", price=12.99),
        ItemAttrs(item_id=2, category="Serum", brand="Neutrogena", price=24.50),
    ]
    result = _format_items(items)
    assert result.count("\n") == 1  # two items → one newline separator


# ── _format_top ───────────────────────────────────────────────────────────────

def test_format_top_empty():
    assert _format_top([]) == "(none)"


def test_format_top_with_data():
    result = _format_top([("Moisturizer", 5), ("Serum", 3)])
    assert "Moisturizer" in result
    assert "x5" in result
    assert "Serum" in result
    assert "x3" in result


# ── FALLBACK_TEXT ─────────────────────────────────────────────────────────────

def test_fallback_text_has_both_languages():
    assert "zh" in FALLBACK_TEXT
    assert "en" in FALLBACK_TEXT
    assert len(FALLBACK_TEXT["zh"]) > 0
    assert len(FALLBACK_TEXT["en"]) > 0


# ── build_structured_prompt ───────────────────────────────────────────────────

def _make_ctx() -> RagContext:
    item = ItemAttrs(item_id=10, category="Lotion", brand="CeraVe", price=9.99)
    target = UserContext(
        user_id=1,
        recent_items=[item],
        top_categories=[("Lotion", 3)],
        top_brands=[("CeraVe", 3)],
    )
    return RagContext(target_user=target, similar_users=[], recommended_items=[item])


def test_structured_prompt_contains_user_id():
    ctx = _make_ctx()
    prompt = build_structured_prompt(ctx, lang="en")
    assert "user_id=1" in prompt


def test_structured_prompt_contains_recommended_item():
    ctx = _make_ctx()
    prompt = build_structured_prompt(ctx, lang="en")
    assert "item_id=10" in prompt


def test_structured_prompt_lang_zh():
    ctx = _make_ctx()
    prompt = build_structured_prompt(ctx, lang="zh")
    assert "繁體中文" in prompt


def test_structured_prompt_no_similar_users():
    ctx = _make_ctx()
    prompt = build_structured_prompt(ctx, lang="en")
    assert "no similar users" in prompt


# ── build_summary_prompt ──────────────────────────────────────────────────────

def test_summary_prompt_includes_structured_output():
    structured = "item_id=10 (brand: CeraVe, category: Lotion): recommended because of brand preference."
    prompt = build_summary_prompt(structured, lang="en")
    assert structured in prompt


def test_summary_prompt_lang_zh():
    prompt = build_summary_prompt("some structured text", lang="zh")
    assert "繁體中文" in prompt
