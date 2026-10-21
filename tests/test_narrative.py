"""Narrator tests: payload, rule-based verdict, citations, escaping, cache key and the streaming fallback."""

from __future__ import annotations

import copy
import json
import re
from datetime import date

import httpx
import numpy as np
import pandas as pd
import pytest

from ga4app import config, llm, narrative
from ga4app.types import Caveats, Filters, Thresholds

TH = Thresholds(views=1000.0, conversion=0.005, views_rule="median")
F = Filters(start=date(2020, 11, 23), end=date(2020, 11, 30), min_views=200)
KEY = "sk-or-v1-testkey0123456789abcdef"

# key, name, category, quadrant, code, views, purchases, cr_adj, confidence, lift, forgone, revenue
_ROWS = [
    ("g1", "Google Cloud Sticker", "Office & Stationery", "Hidden gems", "G1", 500, 8, 0.012, "above", 7.0, 0, 40.0),
    ("g2", "Android Plush [Large]", "Lifestyle & Gifts", "Hidden gems", "G2", 450, 5, 0.010, "uncertain", 5.0, 0, 90.0),
    ("g3", "YouTube Mug", "Drinkware", "Hidden gems", "G3", 600, 6, 0.008, "above", 3.0, 0, 120.0),
    ("w1", "Google Hoodie Navy", "Apparel", "Winners", "W1", 20000, 230, 0.011, "above", 6.0, 0, 12500.5),
    ("w2", "Google Tee White", "Apparel", "Winners", "W2", 15000, 110, 0.007, "above", 2.0, 0, 2400.0),
    ("x1", "Google Campus Bottle", "Drinkware", "Wasted exposure", "X1", 40000, 20, 0.0006, "below", 0, 176.0, 600.0),
    ("x2", "Google Pen Set", "Office & Stationery", "Wasted exposure", "X2", 30000, 50, 0.0017, "below", 0, 99.04, 300.0),
    ("l1", "Google Socks", "Apparel", "Low priority", "", 300, 1, 0.004, "uncertain", 0, 0, 10.0),
    ("n1", "Google Keychain", "Other", "", "", 50, 0, 0.005, "uncertain", 0, 0, 0.0),
]


def make_products() -> pd.DataFrame:
    rows = []
    for key, name, cat, quad, code, v, p, cr_adj, conf, lift, forgone, rev in _ROWS:
        rows.append(
            {
                "product_key": key, "item_name": name, "category_group": cat, "brand": "Google",
                "views": v, "purchases": p, "units": p, "revenue_usd": rev, "promo_views": 0,
                "cr_raw": p / v, "prior_mean": 0.005, "cr_adj": cr_adj,
                "ci_low": cr_adj * 0.6, "ci_high": cr_adj * 1.4, "confidence": conf,
                "eligible": v >= 200, "quadrant": quad, "forgone_purchases": float(forgone),
                "lift_per_1k": float(lift), "rpv": rev / v, "action_rank": int(code[1:]) if code else np.nan,
                "code": code, "views_share": v / 106900, "purchases_exceed_views": False,
            }
        )
    df = pd.DataFrame(rows)
    df["code"] = df["code"].astype("str")
    return df


def make_extras() -> dict:
    period_shift = pd.DataFrame(
        {
            "group": ["All products", "Apparel", "Drinkware"] * 2,
            "period": ["baseline"] * 3 + ["cyber_week"] * 3,
            "period_label": ["Pre-holiday baseline"] * 3 + ["Cyber Week"] * 3,
            "cr": [0.004, 0.005, 0.003, 0.0052, 0.009, 0.0031],
            "lift": [np.nan, np.nan, np.nan, 0.3, 0.8, 0.033],
            "p_value": [np.nan, np.nan, np.nan, 0.001, 0.0004, 0.7],
            "low_sample": [False] * 6,
            "is_base": [True] * 3 + [False] * 3,
        }
    )
    device_lean = pd.DataFrame(
        {"product_key": ["w1", "g1", "x1"], "rr": [2.6, 1.05, 0.5], "lean": ["Mobile-first", "Both", "Desktop-first"]}
    )
    alerts = pd.DataFrame(
        {
            "group_type": ["category", "category"], "group": ["Apparel", "Drinkware"],
            "keywords": ["google hoodie", "water bottle"], "M_ext": [1.6, 1.0], "M_site": [1.02, 1.0],
            "flagged": [True, False], "promote_ahead": [["g2", "l1"], []],
            "promote_names": [["Android Plush [Large]", "Google Socks"], []],
        }
    )
    cannibal = pd.DataFrame(
        {
            "category_group": ["Apparel", "Other", "Apparel"],
            "product_a": ["w1", "l1", "w2"], "product_b": ["g3", "n1", "w1"],
            "item_name_a": ["Google Hoodie Navy", "Google Socks", "Google Tee White"],
            "item_name_b": ["YouTube Mug", "Google Keychain", "Google Hoodie Navy"],
            "n_spike_days": [8, 5, 4], "dip": [0.6, 0.7, 0.1], "q_value": [0.021, 0.01, 0.8],
            "status": ["Likely cannibalized", "Likely cannibalized", "No evidence"],
        }
    )
    device_summary = pd.DataFrame(
        {"device": ["desktop", "mobile", "tablet"], "views": [600, 380, 0], "purchases": [4, 2, 0],
         "cr": [0.0067, 0.0053, np.nan], "views_share": [0.61, 0.39, 0.0]}
    )
    return {"period_shift": period_shift, "device_lean": device_lean, "alerts": alerts, "cannibal": cannibal,
            "device_summary": device_summary}


@pytest.fixture
def payload() -> dict:
    return narrative.build_payload(make_products(), TH, F, window_label="Cyber Week (Nov 23 - Nov 30)", **make_extras())


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.delenv(config.LLM_FAKE_ENV, raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda s: None)


# --------------------------------------------------------------------------- payload


def test_payload_structure_and_rounding(payload):
    json.dumps(payload)  # plain JSON types only
    assert payload["prompt_version"] == narrative.PROMPT_VERSION
    assert [e["code"] for e in payload["winners"]] == ["W1", "W2"]
    assert [e["code"] for e in payload["hidden_gems"]] == ["G1", "G2", "G3"]
    assert [e["code"] for e in payload["wasted_exposure"]] == ["X1", "X2"]
    scope = payload["scope"]
    assert scope["filters"] == F.label()
    assert scope["eligible_products"] == 8 and scope["products_in_scope"] == 9
    assert scope["avg_conversion_pct"] == 0.5
    assert scope["views_threshold"] == 1000 and scope["views_rule"] == "median"
    assert scope["min_views"] == 200 and scope["window"].startswith("Cyber Week")
    assert scope["evidence"] == "ok"
    w1 = payload["winners"][0]
    assert w1["conversion_pct"] == 1.15 and w1["adjusted_conversion_pct"] == 1.1
    assert w1["vs_avg"] == 2.2 and w1["revenue_usd"] == 12500 and w1["views"] == 20000
    assert w1["lift_per_1k"] == 6.0 and "forgone_purchases" not in w1
    x1 = payload["wasted_exposure"][0]
    assert x1["conversion_pct"] == 0.05 and x1["forgone_purchases"] == 176.0
    assert payload["wasted_exposure"][1]["forgone_purchases"] == 99.0
    assert "lift_per_1k" not in x1
    assert payload["hidden_gems"][1]["name"] == "Android Plush (Large)"  # brackets are reserved for codes
    assert payload["caveats"] == list(Caveats().items) == list(narrative.DEFAULT_CAVEATS)


def test_payload_recommendations(payload):
    # "above" before "uncertain", then by adjusted conversion: G1 1.2%, W1 1.1%, G3 0.8% (W2 0.7%, G2 uncertain).
    assert payload["recommended"] == {"feature": ["G1", "W1", "G3"], "retire": ["X1", "X2"]}


def test_payload_optional_sections(payload):
    w1, g1 = payload["winners"][0], payload["hidden_gems"][0]
    assert w1["device_lean"] == "Mobile-first" and w1["mobile_vs_desktop_x"] == 2.6
    assert "device_lean" not in g1  # "Both" is not mentioned
    assert payload["wasted_exposure"][0]["device_lean"] == "Desktop-first"

    shifts = payload["holiday_shift"]
    assert [s["category"] for s in shifts] == ["All products", "Apparel"]  # Drinkware is not significant
    assert shifts[1]["lift_pct"] == 80.0 and shifts[1]["base_conversion_pct"] == 0.5
    assert shifts[1]["focus"] == "Cyber Week" and shifts[1]["base"] == "Pre-holiday baseline"

    (alert,) = payload["outside_signal"]
    assert alert["group"] == "Apparel" and alert["google_change_pct"] == 60 and alert["site_views_change_pct"] == 2
    assert alert["promote_ahead"] == [{"name": "Android Plush (Large)", "code": "G2"}, {"name": "Google Socks"}]

    (pair,) = payload["cannibalization"]  # the uncoded pair and the "No evidence" pair are dropped
    assert pair["spiking_product"] == {"name": "Google Hoodie Navy", "code": "W1"}
    assert pair["affected_product"]["code"] == "G3" and pair["dip_pct"] == 60

    assert payload["device"]["desktop"] == {"conversion_pct": 0.67, "views_share_pct": 61.0}
    assert "tablet" not in payload["device"]


def test_payload_without_optional_inputs():
    payload = narrative.build_payload(make_products(), TH, F, window_label="Full season")
    for key in ("holiday_shift", "device", "outside_signal", "cannibalization"):
        assert key not in payload
    assert "device_lean" not in json.dumps(payload)
    empty = make_extras()
    empty["alerts"] = empty["alerts"].assign(flagged=False)
    empty["cannibal"] = empty["cannibal"].iloc[0:0]
    payload = narrative.build_payload(make_products(), TH, F, window_label="x", **empty)
    assert "outside_signal" not in payload and "cannibalization" not in payload


def test_payload_top_n():
    payload = narrative.build_payload(make_products(), TH, F, window_label="x", top_n=1)
    assert [e["code"] for e in payload["hidden_gems"]] == ["G1"]
    assert payload["recommended"]["feature"] == ["G1", "W1"]
    assert payload["scope"]["evidence"] == "thin"


def test_payload_from_metrics_pipeline():
    """End to end with the real scoring code: every coded product lands in the payload."""
    metrics = pytest.importorskip("ga4app.metrics")
    rng = np.random.default_rng(3)
    keys = [f"p{i:02d}" for i in range(40)]
    views = rng.integers(200, 20000, size=len(keys))
    rates = rng.uniform(0.001, 0.015, size=len(keys))
    purchases = rng.binomial(views, rates)
    daily = pd.DataFrame(
        {
            "event_date": [date(2020, 11, 24)] * len(keys), "device_category": ["desktop"] * len(keys),
            "product_key": keys, "views": views, "add_to_carts": 0, "checkouts": 0, "purchases": purchases,
            "units": purchases, "revenue_usd": purchases * 20.0, "promo_views": 0,
        }
    )
    dim = pd.DataFrame(
        {"product_key": keys, "item_name": [f"Google Item {k}" for k in keys],
         "category_group": ["Apparel", "Drinkware"] * 20, "brand": "Google"}
    )
    products, th = metrics.compute_products(daily, dim, Filters(min_views=100), 500.0)
    payload = narrative.build_payload(products, th, Filters(min_views=100), window_label="Full season")
    coded = {c for c in products["code"].astype("str") if c}
    assert narrative.payload_codes(payload) == coded
    text = narrative.rule_based_verdict(payload)
    assert narrative.unknown_citations(text, payload) == []


# --------------------------------------------------------------------------- rule-based verdict


def test_rule_based_verdict_format_and_citations(payload):
    text = narrative.rule_based_verdict(payload)
    lines = text.splitlines()
    assert lines[0].startswith("**Verdict:** Feature Google Cloud Sticker [G1], Google Hoodie Navy [W1] and YouTube Mug [G3].")
    assert "Retire Google Campus Bottle [X1] from the homepage" in lines[0]
    assert "40,000 views" in lines[0] and "0.05%" in lines[0] and "0.50% average" in lines[0]
    bullets = [line for line in lines if line.startswith("- ")]
    assert 3 <= len(bullets) <= narrative.MAX_BULLETS
    assert lines[-1].startswith("Watch-outs:")
    assert narrative.unknown_citations(text, payload) == []
    cited = set(re.findall(r"\[([A-Z]\d+)\]", text))
    assert cited <= narrative.payload_codes(payload)
    assert {"G1", "W1", "G3", "X1"} <= cited
    assert "Cannibalization" in text and "Outside signal" in text  # flags present -> mentioned
    assert "over the full season" in text
    assert re.search(r"(?<!\\)\$", text) is None  # already md_safe
    assert "- Device lean: Google Hoodie Navy [W1] converts best on mobile;" in text
    assert "desktop" not in text  # X1's lean is not mentioned: it is being retired


def test_rule_based_verdict_is_deterministic(payload):
    text = narrative.rule_based_verdict(payload)
    shuffled = {k: payload[k] for k in reversed(list(payload))}
    shuffled["winners"] = [dict(reversed(list(e.items()))) for e in payload["winners"]]
    assert narrative.rule_based_verdict(copy.deepcopy(payload)) == text
    assert narrative.rule_based_verdict(shuffled) == text
    rebuilt = narrative.build_payload(make_products(), TH, F, window_label="Cyber Week (Nov 23 - Nov 30)", **make_extras())
    assert narrative.rule_based_verdict(rebuilt) == text


def test_rule_based_verdict_without_flags_has_feature_bullets():
    payload = narrative.build_payload(make_products(), TH, F, window_label="x")
    text = narrative.rule_based_verdict(payload)
    bullets = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(bullets) == 3
    assert "extra purchases per 1,000 extra views if new viewers convert like current ones" in bullets[0]
    assert "already earns its traffic" in bullets[1] and "\\$12,500 revenue" in bullets[1]
    assert "Cannibalization" not in text and "Outside signal" not in text and "Device lean" not in text


def test_rule_based_verdict_thin_evidence():
    products = make_products()
    products.loc[products["code"].isin(["G2", "G3", "W1", "W2"]), ["quadrant", "code"]] = ["Low priority", ""]
    payload = narrative.build_payload(products, TH, F, window_label="x")
    assert payload["scope"]["evidence"] == "thin"
    text = narrative.rule_based_verdict(payload)
    assert "evidence is thin" in text and "lower min views" in text.lower()
    assert "[G1]" in text and "[X1]" in text
    assert narrative.unknown_citations(text, payload) == []


def test_rule_based_verdict_with_nothing_qualifying():
    products = make_products().iloc[0:0]
    payload = narrative.build_payload(products, Thresholds(0.0, 0.0), F, window_label="x")
    text = narrative.rule_based_verdict(payload)
    assert text.startswith("**Verdict:** The evidence is thin")
    assert "No product clearly wastes homepage exposure" in text
    assert re.findall(r"\[[A-Z]\d+\]", text) == []


def test_rule_based_verdict_on_hand_built_payload():
    payload = {
        "scope": {"avg_conversion_pct": 0.5, "min_views": 100},
        "hidden_gems": [{"code": "G1", "name": "A", "adjusted_conversion_pct": 1.0, "confidence": "above", "views": 300}],
        "winners": [{"code": "W1", "name": "B", "adjusted_conversion_pct": 2.0, "confidence": "above", "views": 9000}],
        "wasted_exposure": [{"code": "X1", "name": "C", "views": 50000, "conversion_pct": 0.01, "forgone_purchases": 3.21}],
    }
    text = narrative.rule_based_verdict(payload)
    assert "Feature B [W1] and A [G1]" in text and "Retire C [X1]" in text
    assert "about 3.2 purchases forgone" in text


# --------------------------------------------------------------------------- citations, escaping, cache key


def test_unknown_citations(payload):
    text = "Feature [G1] and [Z9]; retire [X1]. Also [W7], [G1, Q3] and [Z9] again. [Google Tee] [1] [g1]"
    assert narrative.unknown_citations(text, payload) == ["Z9", "W7", "Q3"]
    assert narrative.unknown_citations("", payload) == []
    assert narrative.unknown_citations("[G1] [W2] [X2]", payload) == []


def test_md_safe():
    assert narrative.md_safe("costs $5 and $10") == "costs \\$5 and \\$10"
    once = narrative.md_safe("$1,234")
    assert narrative.md_safe(once) == once  # idempotent
    assert narrative.md_safe("") == "" and narrative.md_safe(None) == ""  # type: ignore[arg-type]


def test_cache_key_ignores_dict_ordering(payload, monkeypatch):
    reordered = json.loads(json.dumps(payload, sort_keys=True))
    reordered["scope"] = dict(reversed(list(reordered["scope"].items())))
    key = narrative.cache_key(payload, config.DEFAULT_MODEL)
    assert re.fullmatch(r"[0-9a-f]{64}", key)
    assert narrative.cache_key(reordered, config.DEFAULT_MODEL) == key
    assert narrative.cache_key(payload, "anthropic/claude-haiku-5.5") != key
    changed = copy.deepcopy(payload)
    changed["scope"]["min_views"] = 500
    assert narrative.cache_key(changed, config.DEFAULT_MODEL) != key
    monkeypatch.setattr(narrative, "PROMPT_VERSION", "other")
    assert narrative.cache_key(payload, config.DEFAULT_MODEL) != key


def test_build_messages(payload):
    system, user = narrative.build_messages(payload)
    assert system == {"role": "system", "content": narrative.SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert json.loads(user["content"].split("\n", 1)[1]) == payload
    for phrase in ("Use ONLY the JSON", "purchases / product views", "12 products per page view",
                   "**Verdict:**", "Watch-outs:", "170 words", "evidence is thin", "Mobile-first or Desktop-first"):
        assert phrase in narrative.SYSTEM_PROMPT


# --------------------------------------------------------------------------- verdict_stream


def _sse(*texts: str, model: str = "anthropic/claude-sonnet-5.5", error: dict | None = None) -> httpx.MockTransport:
    def chunk(delta, finish=None, **extra):
        return "data: " + json.dumps({"model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                                      **extra}) + "\n\n"

    body = ": OPENROUTER PROCESSING\n\n" + "".join(chunk({"content": t}) for t in texts)
    if error is not None:
        body += chunk({"content": ""}, "error", error=error)
    else:
        body += chunk({"content": ""}, "stop", usage={"total_tokens": 9}) + "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(200, content=body.encode())

    return httpx.MockTransport(handler)


def _status(status: int) -> httpx.MockTransport:
    def handler(request):
        return httpx.Response(status, json={"error": {"code": status, "message": "Insufficient credits"}})

    return httpx.MockTransport(handler)


def test_verdict_stream_falls_back_on_402(payload):
    sink: dict = {}
    out = "".join(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                           transport=_status(402)))
    assert out.startswith("_AI narrator unavailable (out of OpenRouter credits, HTTP 402). Showing the rule-based verdict._")
    assert out.endswith(narrative.rule_based_verdict(payload))
    assert sink["error"] and "402" in sink["error"] and sink["fallback"] is True
    assert sink["text"] == narrative.rule_based_verdict(payload)
    assert KEY not in out


def test_verdict_stream_fake_transport(payload):
    sink: dict = {}
    chunks = list(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                           transport=llm.fake_transport()))
    assert len(chunks) == 3
    assert "".join(chunks) == "".join(llm.FAKE_TEXT_CHUNKS)
    assert sink["error"] is None and sink["fallback"] is False
    assert sink["model"] == llm.FAKE_MODEL and sink["usage"] == llm.FAKE_USAGE
    assert sink["unknown"] == [] and "[G1]" in sink["text"]


def test_verdict_stream_uses_fake_transport_from_env(payload, monkeypatch):
    monkeypatch.setenv(config.LLM_FAKE_ENV, "1")
    sink: dict = {}
    text = "".join(narrative.verdict_stream(payload, api_key=KEY, model="openrouter/free", sink=sink))
    assert text.startswith("**Verdict:**") and sink["model"] == llm.FAKE_MODEL and sink["error"] is None


def test_verdict_stream_escapes_dollars_and_reports_unknown_codes(payload):
    sink: dict = {}
    out = "".join(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                           transport=_sse("**Verdict:** Feature [G1] ($1,234) ", "and [Q7].")))
    assert "\\$1,234" in out
    assert sink["text"] == "**Verdict:** Feature [G1] ($1,234) and [Q7]."  # raw text for caching
    assert sink["unknown"] == ["Q7"]
    assert sink["model"] == "anthropic/claude-sonnet-5.5" and sink["usage"] == {"total_tokens": 9}


def test_verdict_stream_mid_stream_error(payload):
    sink: dict = {}
    out = "".join(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                           transport=_sse("**Verdict:** Fea", error={"code": "server_error",
                                                                                     "message": "Provider_down"})))
    assert out.startswith("**Verdict:** Fea\n\n_AI narrator unavailable (Provider\\_down).")
    assert out.endswith(narrative.rule_based_verdict(payload))
    assert sink["error"] == "Provider_down" and sink["fallback"] is True


@pytest.mark.parametrize(
    ("api_key", "model", "reason"),
    [(None, config.DEFAULT_MODEL, "no OpenRouter API key"), (KEY, "  ", "no model selected")],
)
def test_verdict_stream_without_key_or_model(payload, api_key, model, reason):
    sink: dict = {}
    out = "".join(narrative.verdict_stream(payload, api_key=api_key, model=model, sink=sink))
    assert reason in out and sink["error"] == reason
    assert out.endswith(narrative.rule_based_verdict(payload))


def test_verdict_stream_empty_answer_falls_back(payload):
    sink: dict = {}
    out = "".join(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                           transport=_sse("   ")))
    assert "returned no text" in sink["error"] and out.endswith(narrative.rule_based_verdict(payload))


def test_verdict_stream_network_error(payload):
    def handler(request):
        raise httpx.ConnectTimeout("slow", request=request)

    sink: dict = {}
    out = "".join(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                           transport=httpx.MockTransport(handler)))
    assert sink["error"] == "request timed out" and "rule-based verdict" in out


def test_verdict_stream_sends_default_fallbacks(payload):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return llm.fake_transport().handle_request(request)

    sink: dict = {}
    list(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, sink=sink,
                                  transport=httpx.MockTransport(handler)))
    body = json.loads(seen[0].content)
    assert body["models"] == list(config.DEFAULT_FALLBACKS)
    assert body["messages"][0]["content"] == narrative.SYSTEM_PROMPT
    assert not {"temperature", "top_p", "top_k"} & set(body)
    list(narrative.verdict_stream(payload, api_key=KEY, model=config.DEFAULT_MODEL, fallbacks=(), sink=sink,
                                  transport=httpx.MockTransport(handler)))
    assert "models" not in json.loads(seen[1].content)
