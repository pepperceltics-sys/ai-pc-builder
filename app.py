import streamlit as st
from urllib.parse import quote_plus
import numpy as np
import json
import hashlib
import time

from recommender import recommend_builds_from_csv_dir

# OpenAI SDK (pip install openai)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

TOP_K_BASE = 200
DISPLAY_TOP = 5
DATA_DIR = "data"
OPENAI_MODEL = "gpt-4o-mini"

# -----------------------------
# Session state (persist results across reruns)
# -----------------------------
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = None
if "shown_builds" not in st.session_state:
    st.session_state.shown_builds = None
if "ai_map" not in st.session_state:
    st.session_state.ai_map = None
if "ai_sig" not in st.session_state:
    st.session_state.ai_sig = None
if "ai_pending" not in st.session_state:
    st.session_state.ai_pending = False
if "ai_next_retry_at" not in st.session_state:
    st.session_state.ai_next_retry_at = 0.0

st.divider()

# -----------------------------
# Helpers
# -----------------------------
def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def clean_str(v) -> str:
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

def show_if_present(label: str, value):
    s = clean_str(value)
    if not s:
        return
    st.write(f"**{label}:** {s}")

def build_search_query(part_name: str, extras: list[str]) -> str:
    base = clean_str(part_name)
    extras_clean = [clean_str(e) for e in extras if clean_str(e)]
    if not base and not extras_clean:
        return ""
    return " ".join([base] + extras_clean).strip()

def google_search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"

def pcpp_search_url(query: str) -> str:
    return f"https://pcpartpicker.com/search/?q={quote_plus(query)}"

def part_link(label: str, part_name: str, extras: list[str], use="google"):
    q = build_search_query(part_name, extras)
    if not q:
        st.caption("Lookup: —")
        return
    url = google_search_url(q) if use == "google" else pcpp_search_url(q)
    st.caption(f"Lookup: [{label}]({url})")

def build_summary_text(build: dict, idx: int) -> str:
    lines = []
    lines.append(f"Build #{idx} — Total: {money(build.get('total_price'))} — Industry: {str(build.get('industry','')).capitalize()}")
    lines.append(f"CPU: {build.get('cpu','—')} ({build.get('cpu_cores','—')} cores, socket {build.get('cpu_socket','—')}) — {money(build.get('cpu_price'))}")
    lines.append(f"GPU: {build.get('gpu','—')} ({build.get('gpu_vram_gb','—')}GB VRAM) — {money(build.get('gpu_price'))}")
    lines.append(f"RAM: {build.get('ram','—')} ({build.get('ram_total_gb','—')}GB, DDR{build.get('ram_ddr','—')}) — {money(build.get('ram_price'))}")
    lines.append(f"Motherboard: {build.get('motherboard','—')} (socket {build.get('mb_socket','—')}, DDR{build.get('mb_ddr','—')}) — {money(build.get('mb_price'))}")
    lines.append(f"PSU: {build.get('psu','—')} ({build.get('psu_wattage','—')}W) — {money(build.get('psu_price'))}")
    lines.append(f"Est. draw: ~{build.get('est_draw_w','—')}W")
    return "\n".join(lines)

def get_part_cols(df):
    candidates = ["cpu", "gpu", "ram", "motherboard", "psu"]
    return [c for c in candidates if c in df.columns]

def is_rate_limit_error(msg: str) -> bool:
    m = (msg or "").lower()
    return ("rate limit" in m) or ("ratelimit" in m) or ("429" in m)

# -----------------------------
# CPU+GPU unique top-N selector
# -----------------------------
def select_diverse_builds(
    df,
    n=5,
    require_unique_cpu_gpu=True,
    part_repeat_penalty=0.0,
    part_cols=None
):
    if df is None or df.empty:
        return df

    part_cols = part_cols or get_part_cols(df)
    rows = df.to_dict(orient="records")

    selected_idx = []
    seen_cpu_gpu = set()
    part_counts = {}

    def norm(v):
        return clean_str(v).lower()

    def cpu_gpu_key(r):
        return (norm(r.get("cpu")), norm(r.get("gpu")))

    def row_parts(r):
        if not part_cols:
            return []
        return [norm(r.get(c)) for c in part_cols if norm(r.get(c))]

    for i, r in enumerate(rows):
        if len(selected_idx) >= n:
            break

        if require_unique_cpu_gpu:
            key = cpu_gpu_key(r)
            if key in seen_cpu_gpu:
                continue

        if part_repeat_penalty and part_cols:
            parts = row_parts(r)
            repeat_score = sum(part_counts.get(p, 0) for p in parts)
            allowed = (len(selected_idx) + 1) * float(part_repeat_penalty)
            if repeat_score > allowed:
                continue

        selected_idx.append(i)
        if require_unique_cpu_gpu:
            seen_cpu_gpu.add(cpu_gpu_key(r))
        for p in row_parts(r):
            part_counts[p] = part_counts.get(p, 0) + 1

    if len(selected_idx) < n:
        for i in range(len(rows)):
            if i in selected_idx:
                continue
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

    return df.iloc[selected_idx].copy()

# -----------------------------
# Slider-based re-ranking
# -----------------------------
def apply_user_weights(df, perf_vs_value: float, include_util: bool):
    if df is None or df.empty:
        return df

    if "perf_score" in df.columns:
        perf_norm = minmax_norm(safe_float_series(df["perf_score"]))
    elif "final_score" in df.columns:
        perf_norm = minmax_norm(safe_float_series(df["final_score"]))
    else:
        perf_norm = np.zeros(len(df))

    if "total_price" in df.columns:
        prices = safe_float_series(df["total_price"])
        inv_price = np.where(np.isfinite(prices) & (prices > 0), 1.0 / prices, 0.0)
        value_norm = minmax_norm(inv_price)
    else:
        value_norm = np.zeros(len(df))

    util_norm = np.zeros(len(df))
    if include_util and "util_score" in df.columns:
        util_norm = minmax_norm(safe_float_series(df["util_score"]))

    w_perf = float(perf_vs_value)
    w_value = 1.0 - w_perf
    w_util = 0.15 if include_util else 0.0

    total_w = w_perf + w_value + w_util
    if total_w <= 1e-12:
        total_w = 1.0
    w_perf /= total_w
    w_value /= total_w
    w_util /= total_w

    out = df.copy()
    out["user_score"] = (w_perf * perf_norm) + (w_value * value_norm) + (w_util * util_norm)
    out = out.sort_values("user_score", ascending=False, kind="mergesort")
    return out

# -----------------------------
# OpenAI: AI pros/cons (rate-limit safe)
# -----------------------------
def get_openai_client():
    if OpenAI is None:
        return None, "OpenAI SDK not installed. Add `openai` to requirements.txt."
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return None, "Missing OPENAI_API_KEY. Add it to Streamlit secrets."
    return OpenAI(api_key=api_key), None

def builds_signature(builds: list[dict], industry: str, budget: float) -> str:
    payload = {"industry": industry, "budget": float(budget), "builds": builds}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

@st.cache_data(show_spinner=False, ttl=3600)
def ai_analyze_builds_cached(sig: str, builds_minimal: list[dict], industry: str, budget: float, model: str):
    client, err = get_openai_client()
    if err:
        return {"__error__": err}

    system = (
        "You are a PC-building assistant. "
        "You will be given up to 5 candidate PC builds with limited fields. "
        "Only use the provided fields; if something isn't provided, treat it as unknown. "
        "Write concise, practical pros/cons for the target industry. "
        "Return JSON ONLY."
    )

    user = {
        "task": "For each build, produce 2 pros and 2 cons. Be concise. Use only provided data.",
        "industry": industry,
        "budget_usd": float(budget),
        "builds": builds_minimal,
        "output_format": [
            {"rank": 1, "pros": ["...","..."], "cons": ["...","..."]}
        ]
    }

    max_tokens = 250  # smaller to reduce TPM pressure
    max_attempts = 4
    base_delay = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)}
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            text = completion.choices[0].message.content or ""

            # Parse strict JSON (allow extra text around it)
            try:
                data = json.loads(text)
            except Exception:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(text[start:end+1])
                else:
                    return {"__error__": "AI response was not valid JSON. Try again."}

            out = {}
            for item in data:
                try:
                    r = int(item.get("rank"))
                    pros = item.get("pros", []) or []
                    cons = item.get("cons", []) or []
                    out[r] = {"pros": pros[:3], "cons": cons[:3]}
                except Exception:
                    continue
            return out

        except Exception as e:
            msg = str(e).lower()
            is_rl = ("ratelimit" in msg) or ("rate limit" in msg) or ("429" in msg)

            if is_rl and attempt < max_attempts:
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue

            if is_rl:
                return {"__error__": "Rate limit hit. Please try again in ~30–60 seconds."}

            return {"__error__": f"OpenAI call failed: {type(e).__name__}"}

    return {"__error__": "Rate limit hit. Please try again shortly."}

def make_builds_minimal_for_ai(shown_builds: list[dict]) -> list[dict]:
    minimal = []
    for i, b in enumerate(shown_builds, start=1):
        minimal.append({
            "rank": i,
            "total_price": b.get("total_price"),
            "est_draw_w": b.get("est_draw_w"),
            "cpu": b.get("cpu"),
            "cpu_cores": b.get("cpu_cores"),
            "cpu_socket": b.get("cpu_socket"),
            "gpu": b.get("gpu"),
            "gpu_vram_gb": b.get("gpu_vram_gb"),
            "ram_total_gb": b.get("ram_total_gb"),
            "ram_ddr": b.get("ram_ddr"),
            "motherboard": b.get("motherboard"),
            "mb_socket": b.get("mb_socket"),
            "mb_ddr": b.get("mb_ddr"),
            "psu": b.get("psu"),
            "psu_wattage": b.get("psu_wattage"),
        })
    return minimal

# -----------------------------
# UI controls
# -----------------------------
with st.expander("Display & ranking options"):
    link_source = st.selectbox("Part lookup links", ["google", "pcpartpicker"], index=0)
    st.caption("Google is more forgiving with imperfect part names; PCPartPicker is nicer when names are exact.")

    st.markdown("### Preference sliders")
    perf_vs_value = st.slider(
        "Value  ⟵───  Performance",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Moves top results toward cheaper/better-value builds (left) or higher performance builds (right)."
    )
    include_util = st.checkbox(
        "Include 'utility' score (if available)",
        value=True,
        help="Keeps a small preference for util_score in the ranking if your recommender provides it."
    )

    st.markdown("### Uniqueness (top 5 variety)")
    make_unique = st.checkbox("Make top 5 builds more unique", value=True)
    require_unique_cpu_gpu = st.checkbox("Require unique CPU + GPU combos", value=True)
    part_repeat_penalty = st.slider(
        "Discourage repeating parts (optional)",
        min_value=0.0, max_value=2.0, value=0.0, step=0.05,
        help="0.0 = only enforce CPU+GPU uniqueness. Higher values allow more repetition."
    )

    st.markdown("### AI explanations")
    use_ai = st.checkbox("Show AI pros/cons for the current top 5", value=False)
    st.caption("If rate-limited, the app will auto-retry and populate AI notes automatically.")

st.divider()

# -----------------------------
# Build card
# -----------------------------
def build_card(build: dict, idx: int, ai_notes: dict | None = None):
    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"Build #{idx}")
        st.caption(f"{build.get('industry', '').capitalize()} build")
    with right:
        st.metric("Total", money(build.get("total_price")))

    parts_left, parts_right = st.columns([2, 2])

    with parts_left:
        st.markdown("**Core components**")

        cpu_name = build.get("cpu", "—")
        st.write(f"**CPU (Model):** {cpu_name} — {money(build.get('cpu_price'))}")
        part_link("CPU", cpu_name, [f"{build.get('cpu_cores','')} cores", f"socket {build.get('cpu_socket','')}"], use=link_source)

        gpu_name = build.get("gpu", "—")
        st.write(f"**GPU (Model):** {gpu_name} — {money(build.get('gpu_price'))}")
        part_link("GPU", gpu_name, [f"{build.get('gpu_vram_gb','')}GB VRAM"], use=link_source)

        ram_name = build.get("ram", "—")
        st.write(f"**RAM (Model):** {ram_name} — {money(build.get('ram_price'))}")
        part_link("RAM", ram_name, [f"{build.get('ram_total_gb','')}GB", f"DDR{build.get('ram_ddr','')}"], use=link_source)

    with parts_right:
        st.markdown("**Platform & power**")

        mb_name = build.get("motherboard", "—")
        st.write(f"**Motherboard (Model):** {mb_name} — {money(build.get('mb_price'))}")
        part_link("Motherboard", mb_name, [f"socket {build.get('mb_socket','')}", f"DDR{build.get('mb_ddr','')}"], use=link_source)

        psu_name = build.get("psu", "—")
        st.write(f"**PSU (Model):** {psu_name} — {money(build.get('psu_price'))}")
        part_link("PSU", psu_name, [f"{build.get('psu_wattage','')}W"], use=link_source)

        st.caption(f"Estimated system draw: ~{build.get('est_draw_w','—')} W")

    # AI Notes
    if ai_notes:
        pros = ai_notes.get("pros", []) or []
        cons = ai_notes.get("cons", []) or []
        if pros or cons:
            st.markdown("**AI Notes**")
            if pros:
                st.markdown("✅ **Pros**")
                for p in pros[:3]:
                    st.write(f"- {p}")
            if cons:
                st.markdown("⚠️ **Cons**")
                for c in cons[:3]:
                    st.write(f"- {c}")

    with st.expander("Copy build summary"):
        summary = build_summary_text(build, idx)
        st.code(summary, language="text")
        st.download_button(
            "Download summary (TXT)",
            data=summary.encode("utf-8"),
            file_name=f"build_{idx}_summary.txt",
            mime="text/plain",
            key=f"dl_summary_{idx}",
        )

    with st.expander("Details (more specs for searching / compatibility)"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**CPU**")
            show_if_present("Socket", build.get("cpu_socket"))
            show_if_present("Cores", build.get("cpu_cores"))
            show_if_present("Brand", build.get("cpu_brand"))
            show_if_present("Series", build.get("cpu_series"))
            show_if_present("Model #", build.get("cpu_model_number"))

        with c2:
            st.markdown("**GPU**")
            show_if_present("VRAM (GB)", build.get("gpu_vram_gb"))
            show_if_present("Brand", build.get("gpu_brand"))
            show_if_present("Chipset", build.get("gpu_chipset"))
            show_if_present("Model #", build.get("gpu_model_number"))

        with c3:
            st.markdown("**Motherboard / RAM / PSU**")
            show_if_present("MB socket", build.get("mb_socket"))
            show_if_present("MB DDR", build.get("mb_ddr"))
            show_if_present("Chipset", build.get("mb_chipset"))
            show_if_present("Form factor", build.get("mb_form_factor"))
            show_if_present("RAM DDR", build.get("ram_ddr"))
            show_if_present("PSU wattage", build.get("psu_wattage"))
            show_if_present("Efficiency", build.get("psu_efficiency"))

    st.divider()

# -----------------------------
# Generate builds (stores results in session_state)
# -----------------------------
if st.button("Generate Builds", type="primary"):
    # Clear AI notes whenever builds change
    st.session_state.ai_map = None
    st.session_state.ai_sig = None
    st.session_state.ai_pending = False
    st.session_state.ai_next_retry_at = 0.0

    with st.spinner("Generating best builds..."):
        df = recommend_builds_from_csv_dir(
            data_dir=DATA_DIR,
            industry=industry,
            total_budget=float(budget),
            top_k=TOP_K_BASE
        )

    if df is None or df.empty:
        st.session_state.ranked_df = None
        st.session_state.shown_builds = None
        st.warning("No compatible builds found under these constraints. Try increasing your budget.")
    else:
        ranked = apply_user_weights(df, perf_vs_value=perf_vs_value, include_util=include_util)

        if make_unique:
            shown_df = select_diverse_builds(
                ranked,
                n=DISPLAY_TOP,
                require_unique_cpu_gpu=require_unique_cpu_gpu,
                part_repeat_penalty=part_repeat_penalty,
                part_cols=get_part_cols(ranked)
            )
        else:
            shown_df = ranked.head(DISPLAY_TOP)

        if "total_price" in shown_df.columns:
            shown_df = shown_df.sort_values("total_price", ascending=True, kind="mergesort")

        st.session_state.ranked_df = ranked
        st.session_state.shown_builds = shown_df.to_dict(orient="records")

# -----------------------------
# Render saved builds + AI (persists across reruns)
# -----------------------------
if st.session_state.shown_builds:
    shown_builds = st.session_state.shown_builds
    ranked = st.session_state.ranked_df

    # AI auto-retry logic (no buttons)
    if use_ai:
        builds_minimal = make_builds_minimal_for_ai(shown_builds)
        sig = builds_signature(builds_minimal, industry, float(budget))
        now = time.time()

        have_notes = (st.session_state.ai_sig == sig) and isinstance(st.session_state.ai_map, dict)

        # If pending and not time yet, show countdown and auto-refresh
        if st.session_state.ai_pending and now < st.session_state.ai_next_retry_at:
            remaining = int(st.session_state.ai_next_retry_at - now)
            st.info(f"AI is temporarily rate-limited. Auto-retrying in {remaining}s…")
            st.autorefresh(interval=5_000, key="ai_autorefresh_waiting")
        else:
            # If we don't have notes yet OR it's time to retry, call OpenAI
            if (not have_notes) or st.session_state.ai_pending:
                with st.spinner("AI is analyzing the top builds..."):
                    ai_map = ai_analyze_builds_cached(sig, builds_minimal, industry, float(budget), OPENAI_MODEL)

                if isinstance(ai_map, dict) and ai_map.get("__error__"):
                    err_msg = ai_map["__error__"]
                    st.warning(f"AI notes unavailable: {err_msg}")

                    if is_rate_limit_error(err_msg):
                        st.session_state.ai_pending = True
                        st.session_state.ai_next_retry_at = time.time() + 60
                        st.session_state.ai_map = None
                        # Do NOT set ai_sig; we want to retry later and succeed
                        st.autorefresh(interval=5_000, key="ai_autorefresh_after_rl")
                    else:
                        # Non-rate-limit error: stop retrying automatically
                        st.session_state.ai_pending = False
                        st.session_state.ai_next_retry_at = 0.0
                        st.session_state.ai_map = None
                        st.session_state.ai_sig = sig
                else:
                    # Success
                    st.session_state.ai_map = ai_map
                    st.session_state.ai_sig = sig
                    st.session_state.ai_pending = False
                    st.session_state.ai_next_retry_at = 0.0

    st.success(
        f"Generated {len(ranked) if ranked is not None else 'some'} ranked builds. "
        f"Showing {len(shown_builds)} build(s) ordered by total price."
    )

    for i, b in enumerate(shown_builds, start=1):
        notes = None
        if use_ai and isinstance(st.session_state.ai_map, dict):
            notes = st.session_state.ai_map.get(i)
        build_card(b, i, ai_notes=notes)

    if ranked is not None:
        st.download_button(
            "Download ranked builds (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K_BASE}_{industry}_{int(budget)}.csv",
            mime="text/csv"
        )
else:
    st.info("Click **Generate Builds** to see recommendations.")
