import streamlit as st
from urllib.parse import quote_plus
import numpy as np
import json
import hashlib
import time

from recommender import recommend_builds_from_csv_dir

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox(
    "Industry",
    ["gaming", "office", "engineering", "content_creation"]
)
budget = st.number_input(
    "Budget (USD)",
    min_value=300,
    max_value=10000,
    value=2000,
    step=50
)

TOP_K_BASE = 200
DISPLAY_TOP = 5
DATA_DIR = "data"
OPENAI_MODEL = "gpt-4o-mini"

st.divider()

# -----------------------------
# Helpers
# -----------------------------
def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def clean_str(v):
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s

def safe_float_series(s):
    return np.array([
        float(x) if str(x).strip().lower() not in ("", "nan", "none") else np.nan
        for x in s
    ])

def minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr)
    mn = np.nanmin(arr[finite])
    mx = np.nanmax(arr[finite])
    if mx - mn < 1e-12:
        out = np.zeros_like(arr)
        out[finite] = 0.5
        out[~finite] = 0.0
        return out
    out = (arr - mn) / (mx - mn)
    out[~finite] = 0.0
    return out

def build_search_query(part_name, extras):
    base = clean_str(part_name)
    extras_clean = [clean_str(e) for e in extras if clean_str(e)]
    return " ".join([base] + extras_clean).strip()

def google_search_url(query):
    return f"https://www.google.com/search?q={quote_plus(query)}"

def part_link(label, part_name, extras):
    q = build_search_query(part_name, extras)
    if q:
        st.caption(f"[{label}]({google_search_url(q)})")

def get_part_cols(df):
    return [c for c in ["cpu", "gpu", "ram", "motherboard", "psu"] if c in df.columns]

# -----------------------------
# Unique CPU+GPU selector
# -----------------------------
def select_diverse_builds(df, n=5):
    rows = df.to_dict(orient="records")
    seen = set()
    idxs = []

    for i, r in enumerate(rows):
        key = (clean_str(r.get("cpu")).lower(), clean_str(r.get("gpu")).lower())
        if key in seen:
            continue
        seen.add(key)
        idxs.append(i)
        if len(idxs) >= n:
            break

    if len(idxs) < n:
        for i in range(len(rows)):
            if i not in idxs:
                idxs.append(i)
            if len(idxs) >= n:
                break

    return df.iloc[idxs].copy()

# -----------------------------
# Ranking
# -----------------------------
def apply_user_weights(df, perf_vs_value):
    if df is None or df.empty:
        return df

    perf = (
        minmax_norm(safe_float_series(df["perf_score"]))
        if "perf_score" in df.columns
        else np.zeros(len(df))
    )

    price = (
        minmax_norm(1.0 / safe_float_series(df["total_price"]))
        if "total_price" in df.columns
        else np.zeros(len(df))
    )

    out = df.copy()
    out["user_score"] = perf_vs_value * perf + (1 - perf_vs_value) * price
    return out.sort_values("user_score", ascending=False, kind="mergesort")

# -----------------------------
# OpenAI helpers
# -----------------------------
def get_openai_client():
    if OpenAI is None:
        return None, "OpenAI SDK not installed"
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if not key:
        return None, "Missing OPENAI_API_KEY"
    return OpenAI(api_key=key), None

def builds_signature(builds, industry, budget):
    payload = {"industry": industry, "budget": budget, "builds": builds}
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

@st.cache_data(show_spinner=False, ttl=3600)
def ai_analyze_builds_cached(sig, builds, industry, budget, model):
    client, err = get_openai_client()
    if err:
        return {"__error__": err}

    system = (
        "You are a PC-building assistant. "
        "Provide concise, practical pros and cons. "
        "Only use provided data."
    )

    user = {
        "industry": industry,
        "budget": budget,
        "builds": builds,
        "format": "JSON list with rank, pros, cons"
    }

    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0.2,
                max_tokens=400,
            )

            text = resp.choices[0].message.content
            data = json.loads(text[text.find("["): text.rfind("]") + 1])

            out = {}
            for item in data:
                out[int(item["rank"])] = {
                    "pros": item.get("pros", [])[:3],
                    "cons": item.get("cons", [])[:3],
                }
            return out

        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                time.sleep(2 ** attempt)
                continue
            return {"__error__": "OpenAI request failed"}

    return {"__error__": "Rate limit hit. Try again shortly."}

# -----------------------------
# UI options
# -----------------------------
with st.expander("Options"):
    perf_vs_value = st.slider("Value ⟵── Performance", 0.0, 1.0, 0.6, 0.05)
    enable_ai = st.checkbox("Enable AI analysis")
    ai_go = st.button("Generate AI explanations") if enable_ai else False

st.divider()

# -----------------------------
# Main
# -----------------------------
if st.button("Generate Builds", type="primary"):
    df = recommend_builds_from_csv_dir(
        DATA_DIR, industry, float(budget), TOP_K_BASE
    )

    ranked = apply_user_weights(df, perf_vs_value)
    shown = select_diverse_builds(ranked, DISPLAY_TOP)
    shown = shown.sort_values("total_price")

    builds = shown.to_dict(orient="records")

    ai_notes = None
    if enable_ai and ai_go:
        minimal = [
            {
                "rank": i + 1,
                "price": b.get("total_price"),
                "cpu": b.get("cpu"),
                "gpu": b.get("gpu"),
                "ram": b.get("ram_total_gb"),
                "psu": b.get("psu_wattage"),
                "draw": b.get("est_draw_w"),
            }
            for i, b in enumerate(builds)
        ]
        sig = builds_signature(minimal, industry, float(budget))
        with st.spinner("AI is analyzing builds..."):
            ai_notes = ai_analyze_builds_cached(
                sig, minimal, industry, float(budget), OPENAI_MODEL
            )
        if isinstance(ai_notes, dict) and ai_notes.get("__error__"):
            st.warning(ai_notes["__error__"])
            ai_notes = None

    for i, b in enumerate(builds, start=1):
        st.subheader(f"Build {i} — {money(b.get('total_price'))}")
        st.write(f"CPU: {b.get('cpu')}")
        st.write(f"GPU: {b.get('gpu')}")
        st.write(f"RAM: {b.get('ram_total_gb')} GB")
        st.write(f"PSU: {b.get('psu_wattage')} W")
        part_link("Search parts", b.get("cpu"), [b.get("gpu")])

        if ai_notes and i in ai_notes:
            st.markdown("**AI Notes**")
            for p in ai_notes[i]["pros"]:
                st.write(f"✅ {p}")
            for c in ai_notes[i]["cons"]:
                st.write(f"⚠️ {c}")

        st.divider()
