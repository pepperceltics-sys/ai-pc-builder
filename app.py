import streamlit as st
from urllib.parse import quote_plus
import numpy as np
from recommender import recommend_builds_from_csv_dir

st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

TOP_K = 50
DISPLAY_TOP = 5
DATA_DIR = "data"

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
    # Best effort conversion for numeric cols
    return np.array([float(x) if str(x).strip().lower() not in ("", "nan", "none") else np.nan for x in s])

def minmax_norm(arr):
    arr = np.array(arr, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr)
    a = arr.copy()
    mn = np.nanmin(a[finite])
    mx = np.nanmax(a[finite])
    if mx - mn < 1e-12:
        out = np.zeros_like(a)
        out[finite] = 0.5
        out[~finite] = 0.0
        return out
    out = (a - mn) / (mx - mn)
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

def select_diverse_builds(df, n=5, max_overlap=1, repeat_penalty=0.35, part_cols=None):
    if df is None or df.empty:
        return df

    part_cols = part_cols or get_part_cols(df)
    if not part_cols:
        return df.head(n)

    selected_idx = []
    part_counts = {}

    rows = df.to_dict(orient="records")

    def row_parts(row):
        parts = []
        for c in part_cols:
            s = clean_str(row.get(c))
            if s:
                parts.append(s)
        return parts

    def overlap_count(a, b):
        return len(set(a).intersection(set(b)))

    for i, r in enumerate(rows):
        if len(selected_idx) == 0:
            selected_idx.append(i)
            for p in row_parts(r):
                part_counts[p] = part_counts.get(p, 0) + 1
            continue

        if len(selected_idx) >= n:
            break

        cand_parts = row_parts(r)

        overlaps = [overlap_count(cand_parts, row_parts(rows[si])) for si in selected_idx]
        worst_overlap = max(overlaps) if overlaps else 0
        if worst_overlap > max_overlap:
            continue

        repeat_score = sum(part_counts.get(p, 0) for p in cand_parts)
        allowed = (len(selected_idx) + 1) * repeat_penalty

        if repeat_score <= allowed:
            selected_idx.append(i)
            for p in cand_parts:
                part_counts[p] = part_counts.get(p, 0) + 1

    if len(selected_idx) < n:
        for i in range(len(rows)):
            if i in selected_idx:
                continue
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

    return df.iloc[selected_idx].copy()

def apply_user_weights(df, perf_vs_value: float, include_util: bool):
    """
    Re-rank builds using slider preferences.
    perf_vs_value: 0.0 (all value) -> 1.0 (all performance)
    include_util: whether to incorporate util_score if available
    """
    if df is None or df.empty:
        return df

    # Performance signal
    perf_norm = None
    if "perf_score" in df.columns:
        perf_norm = minmax_norm(safe_float_series(df["perf_score"]))
    elif "final_score" in df.columns:
        # fallback if perf_score not present
        perf_norm = minmax_norm(safe_float_series(df["final_score"]))
    else:
        perf_norm = np.zeros(len(df))

    # Value signal: cheaper is better, but normalize across candidates
    if "total_price" in df.columns:
        prices = safe_float_series(df["total_price"])
        # avoid divide-by-zero; treat missing prices as bad value
        inv_price = np.where(np.isfinite(prices) & (prices > 0), 1.0 / prices, 0.0)
        value_norm = minmax_norm(inv_price)
    else:
        value_norm = np.zeros(len(df))

    util_norm = np.zeros(len(df))
    if include_util and "util_score" in df.columns:
        util_norm = minmax_norm(safe_float_series(df["util_score"]))

    w_perf = float(perf_vs_value)
    w_value = 1.0 - w_perf

    # Give utility a small constant influence if enabled (keeps “smart” constraints in play)
    w_util = 0.15 if include_util else 0.0

    # Renormalize weights so they sum to 1
    total_w = w_perf + w_value + w_util
    if total_w <= 1e-12:
        total_w = 1.0
    w_perf /= total_w
    w_value /= total_w
    w_util /= total_w

    user_score = w_perf * perf_norm + w_value * value_norm + w_util * util_norm

    out = df.copy()
    out["user_score"] = user_score
    out = out.sort_values("user_score", ascending=False, kind="mergesort")  # stable sort
    return out

# -----------------------------
# UI controls
# -----------------------------
with st.expander("Display & ranking options"):
    link_source = st.selectbox("Part lookup links", ["google", "pcpartpicker"], index=0)
    st.caption("Google is more forgiving with imperfect part names; PCPartPicker is nicer for shopping when names are exact.")

    st.markdown("### Preference sliders")
    perf_vs_value = st.slider(
        "Value  ⟵───  Performance",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Moves top results toward cheaper/better-value builds (left) or higher performance builds (right)."
    )
    include_util = st.checkbox(
        "Include 'utility' score (if available)",
        value=True,
        help="If your recommender provides util_score, this keeps a small preference for that logic in the ranking."
    )

    st.markdown("### Uniqueness (top 5 variety)")
    make_unique = st.checkbox("Make top 5 builds more unique", value=True)
    max_overlap = st.slider(
        "Allowable overlap per build (lower = more unique)",
        min_value=0, max_value=4, value=1,
        help="Limits how many identical parts (CPU/GPU/RAM/MB/PSU) a candidate can share with an already-selected build."
    )
    repeat_penalty = st.slider(
        "Repetition tolerance (lower = fewer repeated parts)",
        min_value=0.0, max_value=2.0, value=0.35, step=0.05,
        help="Soft penalty discouraging the same parts showing up repeatedly across the top 5."
    )

st.divider()

# -----------------------------
# Build card
# -----------------------------
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
        part_link("CPU link", cpu_name, [f"{build.get('cpu_cores','')} cores", f"socket {build.get('cpu_socket','')}"], use=link_source)

        gpu_name = build.get("gpu", "—")
        st.write(f"**GPU (Model):** {gpu_name} — {money(build.get('gpu_price'))}")
        part_link("GPU link", gpu_name, [f"{build.get('gpu_vram_gb','')}GB VRAM"], use=link_source)

        ram_name = build.get("ram", "—")
        st.write(f"**RAM (Model):** {ram_name} — {money(build.get('ram_price'))}")
        part_link("RAM link", ram_name, [f"{build.get('ram_total_gb','')}GB", f"DDR{build.get('ram_ddr','')}"], use=link_source)

    with parts_right:
        st.markdown("**Platform & power**")

        mb_name = build.get("motherboard", "—")
        st.write(f"**Motherboard (Model):** {mb_name} — {money(build.get('mb_price'))}")
        part_link("Motherboard link", mb_name, [f"socket {build.get('mb_socket','')}", f"DDR{build.get('mb_ddr','')}"], use=link_source)

        psu_name = build.get("psu", "—")
        st.write(f"**PSU (Model):** {psu_name} — {money(build.get('psu_price'))}")
        part_link("PSU link", psu_name, [f"{build.get('psu_wattage','')}W"], use=link_source)

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

# -----------------------------
# Main action
# -----------------------------
if st.button("Generate Builds", type="primary"):
    with st.spinner("Generating best builds..."):
        df = recommend_builds_from_csv_dir(
            data_dir=DATA_DIR,
            industry=industry,
            total_budget=float(budget),
            top_k=TOP_K
        )

    if df is None or df.empty:
        st.warning("No compatible builds found under these constraints. Try increasing your budget.")
    else:
        # 1) Apply user preference weights (re-rank)
        ranked = apply_user_weights(df, perf_vs_value=perf_vs_value, include_util=include_util)

        # 2) Pick top 5 (optionally diversified)
        if make_unique:
            shown_df = select_diverse_builds(
                ranked,
                n=DISPLAY_TOP,
                max_overlap=max_overlap,
                repeat_penalty=repeat_penalty,
                part_cols=get_part_cols(ranked)
            )
        else:
            shown_df = ranked.head(DISPLAY_TOP)

        st.success(f"Generated {len(df)} ranked builds. Showing {len(shown_df)} build(s).")

        builds = shown_df.to_dict(orient="records")
        for i, b in enumerate(builds, start=1):
            build_card(b, i)

        st.download_button(
            "Download ranked builds (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K}_{industry}_{int(budget)}.csv",
            mime="text/csv"
        )
