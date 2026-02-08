import streamlit as st
from urllib.parse import quote_plus
import numpy as np
import json
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

st.divider()

# =============================
# Session state (persist results across reruns)
# =============================
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = None
if "shown_builds" not in st.session_state:
    st.session_state.shown_builds = None

# Manual "AI" summary paste box state
if "ai_text_manual" not in st.session_state:
    st.session_state.ai_text_manual = ""


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
    lines.append(
        f"Build #{idx} — Total: {money(build.get('total_price'))} — Industry: {str(build.get('industry','')).capitalize()}"
    )
    lines.append(
        f"CPU: {build.get('cpu','—')} ({build.get('cpu_cores','—')} cores, socket {build.get('cpu_socket','—')}) — {money(build.get('cpu_price'))}"
    )
    lines.append(
        f"GPU: {build.get('gpu','—')} ({build.get('gpu_vram_gb','—')}GB VRAM) — {money(build.get('gpu_price'))}"
    )
    lines.append(
        f"RAM: {build.get('ram','—')} ({build.get('ram_total_gb','—')}GB, DDR{build.get('ram_ddr','—')}) — {money(build.get('ram_price'))}"
    )
    lines.append(
        f"Motherboard: {build.get('motherboard','—')} (socket {build.get('mb_socket','—')}, DDR{build.get('mb_ddr','—')}) — {money(build.get('mb_price'))}"
    )
    lines.append(
        f"PSU: {build.get('psu','—')} ({build.get('psu_wattage','—')}W) — {money(build.get('psu_price'))}"
    )
    lines.append(f"Est. draw: ~{build.get('est_draw_w','—')}W")
    return "\n".join(lines)


def get_part_cols(df):
    candidates = ["cpu", "gpu", "ram", "motherboard", "psu"]
    return [c for c in candidates if c in df.columns]


# =============================
# Uniqueness (CPU+GPU combo hard requirement)
# =============================
def select_diverse_builds(df, n=5, require_unique_cpu_gpu=True, part_repeat_penalty=0.0, part_cols=None):
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


# =============================
# Slider-based re-ranking (+ optional "spend budget" preference)
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


def rank_by_budget_utilization(df, budget_value: float, weight: float = 0.65):
    """
    Push results closer to budget.
    weight=0 -> ignore budget utilization
    weight=1 -> mostly "closest-to-budget" ranking
    """
    if df is None or df.empty or "total_price" not in df.columns:
        return df

    out = df.copy()
    prices = safe_float_series(out["total_price"])
    util = np.where(np.isfinite(prices) & (prices > 0), prices / float(budget_value), 0.0)
    util = np.clip(util, 0.0, 1.2)  # allow slight over for scoring, we filter later
    util_norm = minmax_norm(util)

    # If user_score exists, blend; otherwise just use utilization
    if "user_score" in out.columns:
        score = (1.0 - weight) * minmax_norm(safe_float_series(out["user_score"])) + weight * util_norm
    else:
        score = util_norm

    out["budget_fit_score"] = score
    out = out.sort_values("budget_fit_score", ascending=False, kind="mergesort")
    return out


# =============================
# UI controls
# =============================
with st.expander("Display & ranking options"):
    link_source = st.selectbox("Part lookup links", ["google", "pcpartpicker"], index=0)
    st.caption("Google is more forgiving with imperfect part names; PCPartPicker is nicer when names are exact.")

    st.markdown("### Preference sliders")
    perf_vs_value = st.slider(
        "Value  ⟵───  Performance",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Moves top results toward cheaper/better-value builds (left) or higher performance builds (right).",
    )
    include_util = st.checkbox(
        "Include 'utility' score (if available)",
        value=True,
        help="Keeps a small preference for util_score in the ranking if your recommender provides it.",
    )

    st.markdown("### Use more of the budget")
    spend_weight = st.slider(
        "Prefer builds closer to your budget",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help="Higher values push results closer to the budget instead of picking very cheap value builds.",
    )

    st.markdown("### Uniqueness (top 5 variety)")
    make_unique = st.checkbox("Make top 5 builds more unique", value=True)
    require_unique_cpu_gpu = st.checkbox("Require unique CPU + GPU combos", value=True)
    part_repeat_penalty = st.slider(
        "Discourage repeating parts (optional)",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.05,
        help="0.0 = only enforce CPU+GPU uniqueness. Higher values allow more repetition.",
    )

    st.markdown("### AI commentary (no API)")
    use_manual_ai = st.checkbox("Show/paste AI commentary for the top 5 builds", value=True)
    st.caption("This avoids API rate limits. Generate a summary in ChatGPT and paste it into the box below.")

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
        part_link(
            "CPU",
            cpu_name,
            [f"{build.get('cpu_cores','')} cores", f"socket {build.get('cpu_socket','')}"],
            use=link_source,
        )

        gpu_name = build.get("gpu", "—")
        st.write(f"**GPU (Model):** {gpu_name} — {money(build.get('gpu_price'))}")
        part_link("GPU", gpu_name, [f"{build.get('gpu_vram_gb','')}GB VRAM"], use=link_source)

        ram_name = build.get("ram", "—")
        st.write(f"**RAM (Model):** {ram_name} — {money(build.get('ram_price'))}")
        part_link(
            "RAM",
            ram_name,
            [f"{build.get('ram_total_gb','')}GB", f"DDR{build.get('ram_ddr','')}"],
            use=link_source,
        )

    with parts_right:
        st.markdown("**Platform & power**")

        mb_name = build.get("motherboard", "—")
        st.write(f"**Motherboard (Model):** {mb_name} — {money(build.get('mb_price'))}")
        part_link(
            "Motherboard",
            mb_name,
            [f"socket {build.get('mb_socket','')}", f"DDR{build.get('mb_ddr','')}"],
            use=link_source,
        )

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


# =============================
# Generate builds (stores results in session_state)
# =============================
def call_recommender_with_fallbacks(industry_value: str, budget_value: float, top_k: int):
    """
    Workaround for overly strict or buggy budget logic inside recommender.py:
    - Try requested budget first
    - If empty, retry with larger budgets
    - Always filter results back down to <= original budget in app.py
    """
    fallback_multipliers = [1.0, 1.25, 1.5, 2.0]
    last_df = None

    for m in fallback_multipliers:
        attempt_budget = float(budget_value) * m
        df = recommend_builds_from_csv_dir(
            data_dir=DATA_DIR,
            industry=industry_value,
            total_budget=attempt_budget,
            top_k=top_k,
        )
        last_df = df
        if df is not None and not df.empty:
            # Filter back to original budget (if total_price exists)
            if "total_price" in df.columns:
                prices = safe_float_series(df["total_price"])
                df = df[np.isfinite(prices)]
                df = df[df["total_price"].astype(float) <= float(budget_value)]
            if df is not None and not df.empty:
                return df, m

    return last_df, None


if st.button("Generate Builds", type="primary"):
    with st.spinner("Generating best builds..."):
        df, used_multiplier = call_recommender_with_fallbacks(industry, float(budget), TOP_K_BASE)

    if df is None or df.empty:
        st.session_state.ranked_df = None
        st.session_state.shown_builds = None
        st.warning("No compatible builds found under these constraints. Try increasing your budget or changing industry.")
    else:
        if used_multiplier and used_multiplier > 1.0:
            st.info(
                f"Note: I had to widen the internal search budget to {used_multiplier:.2f}× to find candidates, "
                "then filtered back down to your budget."
            )

        ranked = apply_user_weights(df, perf_vs_value=perf_vs_value, include_util=include_util)
        ranked = rank_by_budget_utilization(ranked, budget_value=float(budget), weight=float(spend_weight))

        if make_unique:
            shown_df = select_diverse_builds(
                ranked,
                n=DISPLAY_TOP,
                require_unique_cpu_gpu=require_unique_cpu_gpu,
                part_repeat_penalty=part_repeat_penalty,
                part_cols=get_part_cols(ranked),
            )
        else:
            shown_df = ranked.head(DISPLAY_TOP)

        # Display order: cheapest -> most expensive (still within top picks)
        if "total_price" in shown_df.columns:
            shown_df = shown_df.sort_values("total_price", ascending=True, kind="mergesort")

        st.session_state.ranked_df = ranked
        st.session_state.shown_builds = shown_df.to_dict(orient="records")


# =============================
# Render saved builds + manual AI summary
# =============================
if st.session_state.shown_builds:
    shown_builds = st.session_state.shown_builds
    ranked = st.session_state.ranked_df

    if use_manual_ai:
        st.markdown("## AI Commentary (Top 5 Builds)")
        st.caption(
            "Generate a short summary in ChatGPT and paste it here. "
            "Markdown is supported, and it will persist during your session."
        )

        st.session_state.ai_text_manual = st.text_area(
            "Paste AI summary",
            value=st.session_state.ai_text_manual,
            height=180,
            placeholder=(
                "Suggested format:\n"
                "- Overall recommendation (2–4 sentences)\n"
                "- Key tradeoffs (3–6 bullets)\n"
                "- Best pick for most people + why\n"
                "- One line per build: 'Build #X best for ...'\n"
            ),
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

    with st.expander("Diagnostics (helps debug budget issues)"):
        if ranked is not None and "total_price" in ranked.columns and not ranked.empty:
            prices = safe_float_series(ranked["total_price"])
            prices = prices[np.isfinite(prices)]
            if prices.size:
                st.write(f"Budget: {money(budget)}")
                st.write(f"Candidate builds returned: {len(ranked)}")
                st.write(f"Min total: {money(prices.min())}")
                st.write(f"Max total: {money(prices.max())}")
                st.write(f"Median total: {money(float(np.median(prices)))}")
else:
    st.info("Click **Generate Builds** to see recommendations.")
