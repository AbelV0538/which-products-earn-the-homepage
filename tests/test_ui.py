"""The verdict card drops the text's leading "**Verdict:**" (its header says it), also while streaming."""

from __future__ import annotations

from ga4app import ui


def test_verdict_md_drops_only_the_lead():
    assert ui._verdict_md("**Verdict:** Feature A [G1].") == "Feature A [G1]."
    assert ui._verdict_md("\n**Verdict:**\nFeature A.") == "Feature A."
    assert ui._verdict_md("Feature A.") == "Feature A."
    # A "**Verdict:**" after other text (e.g. the rule-based fallback after an error notice) stays.
    assert ui._verdict_md("Notice. **Verdict:** Feature A.") == "Notice. **Verdict:** Feature A."


def test_verdict_chunks_strip_a_lead_split_across_chunks():
    assert "".join(ui._verdict_chunks(iter(["**Ver", "dict:**", " Feature", " A [G1]."]))) == "Feature A [G1]."
    assert "".join(ui._verdict_chunks(iter(["\n", "**Verdict:** A."]))) == "A."


def test_verdict_chunks_pass_other_text_through():
    chunks = ["AI narrator unavailable. ", "**Verdict:** Feature A."]
    assert "".join(ui._verdict_chunks(iter(chunks))) == "".join(chunks)
    assert "".join(ui._verdict_chunks(iter(["**Ver"]))) == "**Ver"  # a stream that ends early is kept
    assert list(ui._verdict_chunks(iter([]))) == []
