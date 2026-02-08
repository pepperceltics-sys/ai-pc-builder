import streamlit as st
from urllib.parse import quote_plus
import numpy as np

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

# =============================
# Session state (persist results across reruns)
# =============================
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = None
if "shown_builds" not in st.session_state:
    st.session_state.shown_builds = None
if "ai_text_manual" not in st.session_state:
    st.session_state.ai_text_manual = ""

# Clear cached results when inputs change (prevents stale displays)
params = (industry, float(budget))
if "last_params" not in st.session_state:
    st.session_state.last_params = params
elif st.session_state.last_params != params:
    st.session_state.last_params = params
    st.session_state.ranked_df = None
    st.session_state.shown_builds = None

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
# Uniqueness selector
# - Require unique CPU AND unique GPU across displayed results (if possible)
# =============================
def select_diverse_builds(
    df,
    n=5,
    require_unique_cpu=True,
    require_unique_gpu=True,
    part_cols=None,
):
    if df is None or df.empty:
        return df

    part_cols = part_cols or get_part_cols(df)
    rows = df.to_dict(orient="records")

    selected_idx = []
    seen_cpu = set()
    seen_gpu = set()

    def norm(v):
        return clean_str(v).lower()

    def cpu_key(r):
        return norm(r.get("cpu"))

    def gpu_key(r):
        return norm(r.get("gpu"))

    for i, r in enumerate(rows):
        if len(selected_idx) >= n:
            break

        ck = cpu_key(r)
        gk = gpu_key(r)

        if require_unique_cpu and ck and ck in seen_cpu:
            continue
        if require_unique_gpu and gk and gk in seen_gpu:
            continue

        selected_idx.append(i)
        if require_unique_cpu and ck:
            seen_cpu.add(ck)
        if require_unique_gpu and gk:
            seen_gpu.add(gk)

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
# UI controls (simplified)
# =============================
with st.expander("Options"):
    link_source = st.selectbox("Part lookup links", ["google", "pcpartpicker"], index=0)
    st.caption("Google works best with imperfect names; PCPartPicker is best when names are exact.")

    st.markdown("### Variety in the top 5")
    make_unique = st.checkbox("Make top 5 builds more unique", value=True)
    require_unique_cpu = st.checkbox("Require unique CPUs (top 5)", value=True)
    require_unique_gpu = st.checkbox("Require unique GPUs (top 5)", value=True)

    st.markdown("### Manual AI commentary")
    use_manual_ai = st.checkbox("Show AI commentary box (paste from ChatGPT)", value=True)

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

    # Hard safety filter: never show over-budget builds
    if df is not None and not df.empty and "total_price" in df.columns:
        df = df[df["total_price"].astype(float) <= float(budget)]

    if df is None or df.empty:
        st.session_state.ranked_df = None
        st.session_state.shown_builds = None
        st.warning("No compatible builds found under these constraints. Try increasing your budget.")
    else:
        # Base sort from recommender (final_score descending), then display cheapest->most expensive in top group
        ranked = df.sort_values("final_score", ascending=False, kind="mergesort")

        if make_unique:
            shown_df = select_diverse_builds(
                ranked,
                n=DISPLAY_TOP,
                require_unique_cpu=require_unique_cpu,
                require_unique_gpu=require_unique_gpu,
                part_cols=get_part_cols(ranked),
            )
        else:
            shown_df = ranked.head(DISPLAY_TOP)

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
        st.caption("Paste a comparison from ChatGPT here. This avoids API rate limits.")
        st.session_state.ai_text_manual = st.text_area(
            "Paste AI summary",
            value=st.session_state.ai_text_manual,
            height=180,
            placeholder=(
                "Suggested format:\n"
                "- Overall recommendation (2–4 sentences)\n"
                "- One line per build: best-for + key pros/cons\n"
                "- Any compatibility/balance warnings\n"
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
else:
    st.info("Click **Generate Builds** to see recommendations.")
