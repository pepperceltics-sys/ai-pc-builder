import streamlit as st
from urllib.parse import quote_plus
import numpy as np
import json
import hashlib
import time

from recommender import recommend_builds_from_csv_dir


# =============================
# App config
# =============================
st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

TOP_K_BASE = 1000
DISPLAY_TOP = 5
DATA_DIR = "data"

# AI config (use a cheap/fast model by default)
OPENAI_MODEL = "gpt-4o-mini"
AI_TTL_SECONDS = 24 * 3600

# =============================
# Session state (persist results across reruns)
# =============================
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = None
if "shown_builds" not in st.session_state:
    st.session_state.shown_builds = None

# Manual AI paste box (kept as fallback)
if "ai_text_manual" not in st.session_state:
    st.session_state.ai_text_manual = ""

# Auto AI summary
if "ai_auto_summary" not in st.session_state:
    st.session_state.ai_auto_summary = ""
if "ai_last_sig" not in st.session_state:
    st.session_state.ai_last_sig = ""

# ✅ Clear cached results when inputs change (prevents stale displays)
params = (industry, float(budget))
if "last_params" not in st.session_state:
    st.session_state.last_params = params
elif st.session_state.last_params != params:
    st.session_state.last_params = params
    st.session_state.ranked_df = None
    st.session_state.shown_builds = None
    st.session_state.ai_auto_summary = ""
    st.session_state.ai_last_sig = ""

st.divider()


# =============================
# Helpers
# =============================
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
    return np.array([float(x) if str(x).strip().lower() not in ("", "nan", "none") else np.nan for x in s])


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


# =============================
# Uniqueness selector
# - Require unique CPU AND unique GPU across displayed results (if possible)
# =============================
def select_diverse_builds(
    df,
    n=5,
    require_unique_cpu=True,
    require_unique_gpu=True,
    part_repeat_penalty=0.0,
    part_cols=None,
):
    if df is None or df.empty:
        return df

    part_cols = part_cols or get_part_cols(df)
    rows = df.to_dict(orient="records")

    selected_idx = []
    seen_cpu = set()
    seen_gpu = set()
    part_counts = {}

    def norm(v):
        return clean_str(v).lower()

    def cpu_key(r):
        return norm(r.get("cpu"))

    def gpu_key(r):
        return norm(r.get("gpu"))

    def row_parts(r):
        if not part_cols:
            return []
        return [norm(r.get(c)) for c in part_cols if norm(r.get(c))]

    for i, r in enumerate(rows):
        if len(selected_idx) >= n:
            break

        ck = cpu_key(r)
        gk = gpu_key(r)

        if require_unique_cpu and ck and ck in seen_cpu:
            continue
        if require_unique_gpu and gk and gk in seen_gpu:
            continue

        if part_repeat_penalty and part_cols:
            parts = row_parts(r)
            repeat_score = sum(part_counts.get(p, 0) for p in parts)
            allowed = (len(selected_idx) + 1) * float(part_repeat_penalty)
            if repeat_score > allowed:
                continue

        selected_idx.append(i)

        if require_unique_cpu and ck:
            seen_cpu.add(ck)
        if require_unique_gpu and gk:
            seen_gpu.add(gk)

        for p in row_parts(r):
            part_counts[p] = part_counts.get(p, 0) + 1

    # Fill remaining slots if strict constraints can't find n
    if len(selected_idx) < n:
        for i in range(len(rows)):
            if i in selected_idx:
                continue
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

    return df.iloc[selected_idx].copy()


# =============================
# Slider-based re-ranking (+ budget utilization)
# =============================
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


def rank_by_budget_utilization(df, budget_value: float, weight: float = 0.75):
    if df is None or df.empty or "total_price" not in df.columns:
        return df

    out = df.copy()
    prices = safe_float_series(out["total_price"])
    util = np.where(np.isfinite(prices) & (prices > 0), prices / float(budget_value), 0.0)
    util = np.clip(util, 0.0, 1.0)
    util_norm = minmax_norm(util)

    if "user_score" in out.columns:
        score = (1.0 - weight) * minmax_norm(safe_float_series(out["user_score"])) + weight * util_norm
    else:
        score = util_norm

    out["budget_fit_score"] = score
    return out.sort_values("budget_fit_score", ascending=False, kind="mergesort")


# =============================
# AI prompt + caching
# =============================
def builds_signature(industry: str, budget: float, builds: list[dict]) -> str:
    """
    Stable signature for the displayed top builds.
    If this doesn't change, we reuse cached AI output.
    """
    minimal = []
    for b in builds:
        minimal.append({
            "industry": industry,
            "budget": round(float(budget), 2),
            "total_price": round(float(b.get("total_price", 0.0) or 0.0), 2),
            "cpu": clean_str(b.get("cpu")),
            "gpu": clean_str(b.get("gpu")),
            "ram": clean_str(b.get("ram")),
            "motherboard": clean_str(b.get("motherboard")),
            "psu": clean_str(b.get("psu")),
            "cpu_cores": b.get("cpu_cores"),
            "gpu_vram_gb": b.get("gpu_vram_gb"),
            "ram_total_gb": b.get("ram_total_gb"),
            "psu_wattage": b.get("psu_wattage"),
        })
    payload = json.dumps(minimal, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_ai_prompt(industry: str, budget: float, builds: list[dict]) -> str:
    lines = []
    lines.append("You are a PC building expert. Compare these 5 PC builds for the given use-case and budget.")
    lines.append(f"Use-case/industry: {industry}")
    lines.append(f"Budget: ${budget:,.0f}")
    lines.append("")
    lines.append("Requirements for your answer:")
    lines.append("- Keep it concise and practical.")
    lines.append("- Start with an overall best pick and why (2-4 sentences).")
    lines.append("- Then list each build with: best-for, pros, cons (each in 1 short line).")
    lines.append("- Call out any balance issues (e.g., CPU too weak for GPU, insufficient PSU headroom, RAM mismatch).")
    lines.append("- Prefer builds that spend near the budget if performance gains make sense.")
    lines.append("")
    lines.append("Builds:")
    for i, b in enumerate(builds, start=1):
        lines.append(f"\nBuild #{i} (Total ${float(b.get('total_price',0.0) or 0.0):,.2f})")
        lines.append(f"- CPU: {b.get('cpu','—')} ({b.get('cpu_cores','—')} cores, socket {b.get('cpu_socket','—')})")
        lines.append(f"- GPU: {b.get('gpu','—')} ({b.get('gpu_vram_gb','—')}GB VRAM)")
        lines.append(f"- RAM: {b.get('ram','—')} ({b.get('ram_total_gb','—')}GB, DDR{b.get('ram_ddr','—')})")
        lines.append(f"- Motherboard: {b.get('motherboard','—')} (socket {b.get('mb_socket','—')}, DDR{b.get('mb_ddr','—')})")
        lines.append(f"- PSU: {b.get('psu','—')} ({b.get('psu_wattage','—')}W)")
        lines.append(f"- Est draw: ~{b.get('est_draw_w','—')}W")
    return "\n".join(lines)


def get_openai_api_key() -> str:
    # Try common secret keys (you said it's already in Streamlit secrets)
    for k in ["OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY", "openai_key"]:
        if k in st.secrets and str(st.secrets[k]).strip():
            return str(st.secrets[k]).strip()
    return ""


@st.cache_data(show_spinner=False, ttl=AI_TTL_SECONDS)
def ai_summary_cached(signature: str, prompt: str, model: str) -> str:
    """
    Cached AI call by signature. Long TTL reduces rate-limit pain.
    """
    api_key = get_openai_api_key()
    if not api_key:
        return "AI summary unavailable: missing OpenAI API key in Streamlit secrets."

    # Import OpenAI SDK lazily
    try:
        from openai import OpenAI
        from openai import RateLimitError, APIError, APITimeoutError
    except Exception:
        return "AI summary unavailable: OpenAI SDK not installed. Add `openai` to requirements.txt."

    client = OpenAI(api_key=api_key)

    # Light retry/backoff on rate limit or transient API errors
    delays = [1.0, 2.0, 4.0]
    for attempt in range(len(delays) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that gives succinct PC build advice."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )
            text = resp.choices[0].message.content or ""
            return text.strip() if text.strip() else "AI summary returned empty text."
        except (RateLimitError, APITimeoutError, APIError) as e:
            if attempt >= len(delays):
                # Return a friendly message (and do NOT crash app)
                return f"AI notes unavailable: rate limit or transient API error. Please try again later.\n\nDetails: {type(e).__name__}"
            time.sleep(delays[attempt])
        except Exception as e:
            return f"AI notes unavailable due to an unexpected error: {type(e).__name__}"


# =============================
# UI controls
# =============================
with st.expander("Display & ranking options"):
    link_source = st.selectbox("Part lookup links", ["google", "pcpartpicker"], index=0)

    st.markdown("### Preference sliders")
    perf_vs_value = st.slider("Value  ⟵───  Performance", 0.0, 1.0, 0.6, 0.05)
    include_util = st.checkbox("Include 'utility' score (if available)", value=True)

    st.markdown("### Spend closer to budget")
    spend_weight = st.slider("Prefer builds closer to your budget", 0.0, 1.0, 0.75, 0.05)

    st.markdown("### Uniqueness (top 5 variety)")
    make_unique = st.checkbox("Make top 5 builds more unique", value=True)
    require_unique_cpu = st.checkbox("Require unique CPUs (top 5)", value=True)
    require_unique_gpu = st.checkbox("Require unique GPUs (top 5)", value=True)
    part_repeat_penalty = st.slider("Discourage repeating parts (optional)", 0.0, 2.0, 0.0, 0.05)

    st.markdown("### AI commentary")
    auto_ai = st.checkbox("Auto-generate AI comparison (cached)", value=True)
    st.caption("Uses OpenAI once per unique top-5 set, then reuses cached output for ~24 hours.")
    use_manual_ai = st.checkbox("Manual paste fallback box", value=False)

st.divider()


# =============================
# Build card
# =============================
def build_card(build: dict, idx: int):
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
        with c2:
            st.markdown("**GPU**")
            show_if_present("VRAM (GB)", build.get("gpu_vram_gb"))
        with c3:
            st.markdown("**Motherboard / RAM / PSU**")
            show_if_present("MB socket", build.get("mb_socket"))
            show_if_present("MB DDR", build.get("mb_ddr"))
            show_if_present("RAM DDR", build.get("ram_ddr"))
            show_if_present("PSU wattage", build.get("psu_wattage"))

    st.divider()


# =============================
# Generate builds
# =============================
if st.button("Generate Builds", type="primary"):
    with st.spinner("Generating best builds..."):
        df = recommend_builds_from_csv_dir(
            data_dir=DATA_DIR,
            industry=industry,
            total_budget=float(budget),
            top_k=TOP_K_BASE,
        )

    # Hard safety filter (never show over-budget builds)
    if df is not None and not df.empty and "total_price" in df.columns:
        df = df[df["total_price"].astype(float) <= float(budget)]

    if df is None or df.empty:
        st.session_state.ranked_df = None
        st.session_state.shown_builds = None
        st.session_state.ai_auto_summary = ""
        st.session_state.ai_last_sig = ""
        st.warning(
            "No compatible builds found under these constraints at this budget. "
            "Try increasing budget OR switching industry, OR your dataset may not have parts that meet the minimum requirements."
        )
    else:
        ranked = apply_user_weights(df, perf_vs_value=perf_vs_value, include_util=include_util)
        ranked = rank_by_budget_utilization(ranked, budget_value=float(budget), weight=float(spend_weight))

        if make_unique:
            shown_df = select_diverse_builds(
                ranked,
                n=DISPLAY_TOP,
                require_unique_cpu=require_unique_cpu,
                require_unique_gpu=require_unique_gpu,
                part_repeat_penalty=part_repeat_penalty,
                part_cols=get_part_cols(ranked),
            )
        else:
            shown_df = ranked.head(DISPLAY_TOP)

        # Display order: cheapest -> most expensive (within selected top group)
        if "total_price" in shown_df.columns:
            shown_df = shown_df.sort_values("total_price", ascending=True, kind="mergesort")

        st.session_state.ranked_df = ranked
        st.session_state.shown_builds = shown_df.to_dict(orient="records")

        # ✅ AUTO AI: compute only if enabled, and only when build signature changes
        if auto_ai and st.session_state.shown_builds:
            sig = builds_signature(industry, float(budget), st.session_state.shown_builds)
            if sig != st.session_state.ai_last_sig:
                prompt = build_ai_prompt(industry, float(budget), st.session_state.shown_builds)
                with st.spinner("Generating AI comparison (cached)..."):
                    st.session_state.ai_auto_summary = ai_summary_cached(sig, prompt, OPENAI_MODEL)
                st.session_state.ai_last_sig = sig


# =============================
# Render saved builds + AI summary
# =============================
if st.session_state.shown_builds:
    shown_builds = st.session_state.shown_builds
    ranked = st.session_state.ranked_df

    # Auto AI summary section
    if auto_ai:
        st.markdown("## AI Comparison (Top 5 Builds)")
        if st.session_state.ai_auto_summary:
            st.markdown(st.session_state.ai_auto_summary)
        else:
            st.caption("AI comparison will appear here after generation.")

        st.divider()

    # Manual fallback paste box (optional)
    if use_manual_ai:
        st.markdown("## Manual AI Commentary (Fallback)")
        st.caption("If the API ever rate-limits, you can paste a summary here.")
        st.session_state.ai_text_manual = st.text_area(
            "Paste AI summary",
            value=st.session_state.ai_text_manual,
            height=180,
        ).strip()
        if st.session_state.ai_text_manual:
            st.markdown(st.session_state.ai_text_manual)
            st.divider()

    st.success(
        f"Generated {len(ranked) if ranked is not None else 'some'} ranked builds. "
        f"Showing {len(shown_builds)} build(s) ordered by total price."
    )

    for i, b in enumerate(shown_builds, start=1):
        build_card(b, i)

    if ranked is not None:
        st.download_button(
            "Download ranked builds (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K_BASE}_{industry}_{int(budget)}.csv",
            mime="text/csv",
        )
else:
    st.info("Click **Generate Builds** to see recommendations.")
