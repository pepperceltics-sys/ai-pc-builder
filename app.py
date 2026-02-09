import streamlit as st
from urllib.parse import quote_plus

from recommender import recommend_builds_from_csv_dir


# =============================
# App config
# =============================
st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Pick a use-case and budget. We’ll generate beginner-friendly PC builds from your CSV dataset.")

industry = st.selectbox("Use-case", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

TOP_K_BASE = 1000
DISPLAY_TOP = 5
DATA_DIR = "data"

# =============================
# Session state
# =============================
if "ranked_df" not in st.session_state:
    st.session_state.ranked_df = None
if "shown_builds" not in st.session_state:
    st.session_state.shown_builds = None
if "ai_text_manual" not in st.session_state:
    st.session_state.ai_text_manual = ""

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


def google_search_url(query: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(query)}"


def pcpp_search_url(query: str) -> str:
    return f"https://pcpartpicker.com/search/?q={quote_plus(query)}"


def build_search_query(*parts) -> str:
    """
    Builds a clean, specific search query.
    Ignores empty/unknown values and deduplicates tokens lightly.
    """
    tokens = []
    for p in parts:
        s = clean_str(p)
        if not s:
            continue
        s_up = s.upper()
        if s_up in ("UNKNOWN", "N/A", "NA"):
            continue
        tokens.append(s)

    # light dedupe while preserving order
    seen = set()
    out = []
    for t in tokens:
        key = t.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t.strip())

    return " ".join(out).strip()


def part_links_row(query: str):
    if not query:
        st.caption("Lookup: —")
        return
    g = google_search_url(query)
    p = pcpp_search_url(query)
    st.caption(f"Lookup: [Google]({g})  |  [PCPartPicker]({p})")


# =============================
# Beginner checks + summary (unchanged)
# =============================
def compute_checks(build: dict, budget_value: float) -> dict:
    cpu_socket = clean_str(build.get("cpu_socket")).lower()
    mb_socket = clean_str(build.get("mb_socket")).lower()

    ram_ddr = clean_str(build.get("ram_ddr")).upper()
    mb_ddr = clean_str(build.get("mb_ddr")).upper()

    ram_modules = int(build.get("ram_modules") or 0)
    mb_slots = int(build.get("mb_ram_slots") or 0)

    est_draw = float(build.get("est_draw_w") or 0)
    psu_w = float(build.get("psu_wattage") or 0)
    headroom_pct = float(build.get("psu_headroom_pct") or 0.0)

    total = float(build.get("total_price") or 0.0)
    used_pct = (total / float(budget_value)) if float(budget_value) > 0 else 0.0
    leftover = float(budget_value) - total

    return {
        "socket_match": bool(cpu_socket and mb_socket and cpu_socket == mb_socket),
        "ddr_match": bool(ram_ddr and mb_ddr and ram_ddr != "UNKNOWN" and mb_ddr != "UNKNOWN" and ram_ddr == mb_ddr),
        "ram_slots_ok": bool(ram_modules > 0 and mb_slots > 0 and ram_modules <= mb_slots),
        "psu_ok": bool(psu_w > 0 and est_draw > 0 and psu_w >= est_draw),
        "psu_headroom_pct": headroom_pct,
        "est_draw_w": est_draw,
        "psu_w": psu_w,
        "budget_used_pct": used_pct,
        "budget_leftover": leftover,
        "over_budget": total > float(budget_value) + 1e-6,
    }


def performance_tier(industry: str, cpu_cores: int, gpu_vram: float, ram_gb: int) -> str:
    industry = str(industry)
    if industry == "office":
        if ram_gb >= 16 and cpu_cores >= 6:
            return "Strong everyday + multitasking"
        return "Solid everyday use"
    if industry == "gaming":
        if gpu_vram >= 16 and cpu_cores >= 8:
            return "High-end gaming tier"
        if gpu_vram >= 12 and cpu_cores >= 6:
            return "Great 1440p/High settings tier"
        if gpu_vram >= 8:
            return "Good 1080p/High settings tier"
        return "Entry gaming tier"
    if industry == "engineering":
        if ram_gb >= 64 and cpu_cores >= 12:
            return "Heavy CAD/simulation tier"
        if ram_gb >= 32 and cpu_cores >= 8:
            return "Strong CAD + productivity tier"
        return "Entry engineering tier"
    if industry == "content_creation":
        if ram_gb >= 64 and gpu_vram >= 16:
            return "4K+ editing / heavy creation tier"
        if ram_gb >= 32 and gpu_vram >= 12:
            return "Great editing/creation tier"
        return "Entry creation tier"
    return "General tier"


def beginner_summary(build: dict, checks: dict) -> list[str]:
    cpu_cores = int(build.get("cpu_cores") or 0)
    gpu_vram = float(build.get("gpu_vram_gb") or 0.0)
    ram_gb = int(build.get("ram_total_gb") or 0)

    total = float(build.get("total_price") or 0.0)
    used_pct = checks["budget_used_pct"] * 100.0
    leftover = checks["budget_leftover"]

    tier = performance_tier(build.get("industry", ""), cpu_cores, gpu_vram, ram_gb)

    bullets = []
    bullets.append(f"**What it’s best for:** {tier}.")
    bullets.append(f"**Budget fit:** Uses ~{used_pct:.0f}% of your budget ({money(total)}), leftover {money(leftover)}.")

    gpu_price = float(build.get("gpu_price") or 0.0)
    cpu_price = float(build.get("cpu_price") or 0.0)
    gpu_share = (gpu_price / total) if total > 0 else 0.0
    cpu_share = (cpu_price / total) if total > 0 else 0.0

    if build.get("industry") == "gaming":
        if gpu_share >= 0.55:
            bullets.append("**Why it’s good:** GPU-heavy (great for gaming FPS).")
        elif cpu_share >= 0.40:
            bullets.append("**Watch out:** CPU-heavy relative to GPU (might limit gaming FPS at higher resolutions).")
        else:
            bullets.append("**Why it’s good:** Balanced CPU/GPU spending for gaming.")
    elif build.get("industry") in ("engineering", "content_creation"):
        if ram_gb < 32:
            bullets.append("**Watch out:** RAM is under 32GB—many creation/engineering workloads benefit from 32GB+.")
        if cpu_cores < 8:
            bullets.append("**Watch out:** CPU is under 8 cores—some workloads may feel slower.")
        if gpu_vram < 10 and build.get("industry") == "content_creation":
            bullets.append("**Watch out:** VRAM is modest—effects-heavy editing can benefit from more VRAM.")
    else:
        if ram_gb < 16:
            bullets.append("**Upgrade path:** Consider moving to 16GB RAM for smoother multitasking.")
        else:
            bullets.append("**Why it’s good:** Plenty of RAM for everyday multitasking.")

    hr = checks["psu_headroom_pct"]
    if hr < 0.15:
        bullets.append("**Watch out:** PSU headroom is tight (<15%). A bit more wattage is safer for future upgrades.")
    elif hr < 0.30:
        bullets.append("**Good:** PSU headroom is reasonable (~15–30%).")
    else:
        bullets.append("**Great:** PSU has healthy headroom (30%+), good for upgrades.")

    if not (checks["socket_match"] and checks["ddr_match"] and checks["ram_slots_ok"]):
        bullets.append("**Warning:** One or more compatibility checks is failing—double check before buying parts.")

    if build.get("industry") == "gaming":
        bullets.append("**Next upgrade idea:** GPU first (biggest gaming gains).")
    elif build.get("industry") == "content_creation":
        bullets.append("**Next upgrade idea:** RAM or GPU (depends on editing apps and codec).")
    elif build.get("industry") == "engineering":
        bullets.append("**Next upgrade idea:** RAM (often the first bottleneck).")
    else:
        bullets.append("**Next upgrade idea:** SSD/RAM (not included in this dataset).")

    return bullets


# =============================
# Uniqueness selector: unique CPU and unique GPU if possible
# =============================
def select_diverse_builds(df, n=5, require_unique_cpu=True, require_unique_gpu=True):
    if df is None or df.empty:
        return df

    rows = df.to_dict(orient="records")
    selected_idx = []
    seen_cpu = set()
    seen_gpu = set()

    def norm(v):
        return clean_str(v).lower()

    for i, r in enumerate(rows):
        if len(selected_idx) >= n:
            break

        ck = norm(r.get("cpu"))
        gk = norm(r.get("gpu"))

        if require_unique_cpu and ck and ck in seen_cpu:
            continue
        if require_unique_gpu and gk and gk in seen_gpu:
            continue

        selected_idx.append(i)
        if require_unique_cpu and ck:
            seen_cpu.add(ck)
        if require_unique_gpu and gk:
            seen_gpu.add(gk)

    if len(selected_idx) < n:
        for i in range(len(rows)):
            if i in selected_idx:
                continue
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

    return df.iloc[selected_idx].copy()


# =============================
# Options
# =============================
with st.expander("Options"):
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
def build_card(build: dict, idx: int, budget_value: float):
    checks = compute_checks(build, budget_value)

    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"Build #{idx}")
        st.caption(f"{build.get('industry', '').replace('_',' ').title()} build")
    with right:
        st.metric("Total", money(build.get("total_price")))

    # --- Budget utilization
    used_pct = max(0.0, min(1.0, checks["budget_used_pct"]))
    st.markdown("**Budget utilization**")
    st.progress(used_pct)
    st.caption(f"Uses {money(build.get('total_price'))} of {money(budget_value)} • Leftover: {money(checks['budget_leftover'])}")

    # --- Compatibility badges
    st.markdown("**Compatibility checks**")
    b1, b2, b3, b4 = st.columns(4)

    def badge(col, ok: bool, label_ok: str, label_bad: str, help_text: str):
        with col:
            if ok:
                st.success(label_ok)
            else:
                st.warning(label_bad)
            st.caption(help_text)

    badge(b1, checks["socket_match"], "✅ CPU socket", "⚠️ CPU socket", "CPU and motherboard socket must match.")
    badge(b2, checks["ddr_match"], "✅ RAM type", "⚠️ RAM type", "DDR version (DDR4/DDR5) must match motherboard.")
    badge(b3, checks["ram_slots_ok"], "✅ RAM slots", "⚠️ RAM slots", "RAM kit must fit available DIMM slots.")

    psu_headroom_ok = checks["psu_ok"] and checks["psu_headroom_pct"] >= 0.15
    badge(b4, psu_headroom_ok, "✅ PSU headroom", "⚠️ PSU headroom", "Aim for ~15–30%+ headroom for upgrades.")

    if checks["psu_ok"]:
        st.caption(f"Estimated draw: ~{int(checks['est_draw_w'])}W • PSU: {int(checks['psu_w'])}W • Headroom: {checks['psu_headroom_pct']*100:.0f}%")

    # --- Beginner summary
    st.markdown("### Beginner Summary (why this build makes sense)")
    for b in beginner_summary(build, checks):
        st.write(f"- {b}")

    st.divider()

    # --- Parts
    parts_left, parts_right = st.columns([2, 2])

    with parts_left:
        st.markdown("**Core components**")

        # CPU search query (more specific)
        cpu_query = build_search_query(
            build.get("cpu_brand"),
            build.get("cpu"),
            build.get("cpu_model_number"),
            f"{build.get('cpu_cores','')} core",
            build.get("cpu_socket"),
        )
        st.write(f"**CPU:** {build.get('cpu','—')} — {money(build.get('cpu_price'))}")
        part_links_row(cpu_query)

        # GPU search query (chipset/model number/VRAM help a lot)
        gpu_query = build_search_query(
            build.get("gpu_brand"),
            build.get("gpu_chipset"),
            build.get("gpu"),
            build.get("gpu_model_number"),
            f"{build.get('gpu_vram_gb','')}GB",
        )
        st.write(f"**GPU:** {build.get('gpu','—')} — {money(build.get('gpu_price'))}")
        part_links_row(gpu_query)

        # RAM search query
        ram_query = build_search_query(
            build.get("ram_brand"),
            build.get("ram"),
            build.get("ram_model_number"),
            f"{build.get('ram_total_gb','')}GB",
            f"DDR{build.get('ram_ddr','')}",
            f"{build.get('ram_modules','')}x",
        )
        st.write(f"**RAM:** {build.get('ram','—')} — {money(build.get('ram_price'))}")
        part_links_row(ram_query)

    with parts_right:
        st.markdown("**Platform & power**")

        # Motherboard search query
        mb_query = build_search_query(
            build.get("motherboard"),
            build.get("mb_model_number"),
            build.get("mb_chipset"),
            build.get("mb_form_factor"),
            build.get("mb_socket"),
            f"DDR{build.get('mb_ddr','')}",
        )
        st.write(f"**Motherboard:** {build.get('motherboard','—')} — {money(build.get('mb_price'))}")
        part_links_row(mb_query)

        # PSU search query
        psu_query = build_search_query(
            build.get("psu_brand"),
            build.get("psu"),
            build.get("psu_model_number"),
            f"{build.get('psu_wattage','')}W",
            build.get("psu_efficiency"),
        )
        st.write(f"**PSU:** {build.get('psu','—')} — {money(build.get('psu_price'))}")
        part_links_row(psu_query)

        st.caption(f"Estimated system draw: ~{build.get('est_draw_w','—')} W")

    with st.expander("Copy build summary"):
        summary = (
            f"Build #{idx}\n"
            f"Industry: {build.get('industry','')}\n"
            f"Total: {money(build.get('total_price'))}\n\n"
            f"CPU: {build.get('cpu','—')} — {money(build.get('cpu_price'))}\n"
            f"GPU: {build.get('gpu','—')} — {money(build.get('gpu_price'))}\n"
            f"RAM: {build.get('ram','—')} — {money(build.get('ram_price'))}\n"
            f"Motherboard: {build.get('motherboard','—')} — {money(build.get('mb_price'))}\n"
            f"PSU: {build.get('psu','—')} — {money(build.get('psu_price'))}\n"
        )
        st.code(summary, language="text")
        st.download_button(
            "Download summary (TXT)",
            data=summary.encode("utf-8"),
            file_name=f"build_{idx}_summary.txt",
            mime="text/plain",
            key=f"dl_summary_{idx}",
        )

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

    # Safety: never show over-budget builds
    if df is not None and not df.empty and "total_price" in df.columns:
        df = df[df["total_price"].astype(float) <= float(budget)]

    if df is None or df.empty:
        st.session_state.ranked_df = None
        st.session_state.shown_builds = None
        st.warning("No compatible builds found under these constraints. Try increasing your budget.")
    else:
        ranked = df.sort_values("final_score", ascending=False, kind="mergesort")

        if make_unique:
            shown_df = select_diverse_builds(
                ranked,
                n=DISPLAY_TOP,
                require_unique_cpu=require_unique_cpu,
                require_unique_gpu=require_unique_gpu,
            )
        else:
            shown_df = ranked.head(DISPLAY_TOP)

        # Display cheapest -> most expensive in selected top group
        if "total_price" in shown_df.columns:
            shown_df = shown_df.sort_values("total_price", ascending=True, kind="mergesort")

        st.session_state.ranked_df = ranked
        st.session_state.shown_builds = shown_df.to_dict(orient="records")


# =============================
# Render builds + optional manual AI paste box
# =============================
if st.session_state.shown_builds:
    shown_builds = st.session_state.shown_builds
    ranked = st.session_state.ranked_df

    if use_manual_ai:
        st.markdown("## AI Commentary (Top 5 Builds)")
        st.caption("Paste a comparison from ChatGPT here. (No API calls = no rate limits.)")
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
        build_card(b, i, float(budget))

    if ranked is not None:
        st.download_button(
            "Download ranked builds (CSV)",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K_BASE}_{industry}_{int(budget)}.csv",
            mime="text/csv",
        )
else:
    st.info("Click **Generate Builds** to see recommendations.")
