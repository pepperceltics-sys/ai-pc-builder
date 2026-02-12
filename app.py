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

# --- UI polish (no logic changes)
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
h1 {margin-bottom: 0.25rem;}
div[data-testid="stMetricValue"] {font-size: 1.4rem;}
pre {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([3, 2], vertical_alignment="center")
with top_left:
    st.markdown("### Build a PC that makes sense (even if you're new)")
    st.caption("We’ll generate compatible builds near your budget and explain what to upgrade next.")
with top_right:
    st.info("Tip: If your leftover budget is large, your dataset may be missing mid-priced parts.", icon="💡")

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


# =============================
# Search providers (fixed + expanded)
# =============================
def google_url(q):        return f"https://www.google.com/search?q={quote_plus(q)}"
def pcpp_url(q):          return f"https://pcpartpicker.com/search/?q={quote_plus(q)}"
def bestbuy_url(q):       return f"https://www.bestbuy.com/site/searchpage.jsp?st={quote_plus(q)}"
def amazon_url(q):        return f"https://www.amazon.com/s?k={quote_plus(q)}"
def newegg_url(q):        return f"https://www.newegg.com/p/pl?d={quote_plus(q)}"
def microcenter_url(q):   return f"https://www.microcenter.com/search/search_results.aspx?Ntt={quote_plus(q)}"
def bh_url(q):            return f"https://www.bhphotovideo.com/c/search?Ntt={quote_plus(q)}"
def ebay_url(q):          return f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(q)}"
def google_shop_url(q):   return f"https://www.google.com/search?tbm=shop&q={quote_plus(q)}"

SEARCH_PROVIDERS = {
    "google": google_url,
    "pcpartpicker": pcpp_url,
    "bestbuy": bestbuy_url,
    "amazon": amazon_url,
    "newegg": newegg_url,
    "microcenter": microcenter_url,
    "bhphoto": bh_url,
    "googleshopping": google_shop_url,
    "ebay": ebay_url,
}

def part_link(label: str, part_name: str, extras: list[str], use="google"):
    q = build_search_query(part_name, extras)
    if not q:
        st.caption("Lookup: —")
        return
    use_key = clean_str(use).lower()
    fn = SEARCH_PROVIDERS.get(use_key, google_url)
    url = fn(q)
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
# Beginner checks + summary
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

    checks = {
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
    return checks


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

    if len(selected_idx) < n:
        for i in range(len(rows)):
            if i in selected_idx:
                continue
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

    return df.iloc[selected_idx].copy()


# =============================
# UI controls (cleaner)
# =============================
with st.expander("⚙️ Options", expanded=False):
    st.markdown("#### Part lookup destination")
    link_source = st.selectbox(
        "Open links in…",
        ["google", "pcpartpicker", "bestbuy", "amazon", "newegg", "microcenter", "bhphoto", "googleshopping", "ebay"],
        index=0,
        label_visibility="collapsed",
    )
    st.caption("Google works best with imperfect names; PCPartPicker is best when names are exact.")

    st.divider()
    st.markdown("#### Variety in the top 5")
    c1, c2, c3 = st.columns(3)
    with c1:
        make_unique = st.checkbox("Make top 5 more unique", value=True)
    with c2:
        require_unique_cpu = st.checkbox("Unique CPUs", value=True)
    with c3:
        require_unique_gpu = st.checkbox("Unique GPUs", value=True)

    st.divider()
    st.markdown("#### Manual AI commentary")
    use_manual_ai = st.checkbox("Show AI commentary box (paste from ChatGPT)", value=True)

st.divider()


# =============================
# Build card (styled + tabs)
# =============================
def build_card(build: dict, idx: int, budget_value: float):
    checks = compute_checks(build, budget_value)

    with st.container(border=True):
        header_l, header_r = st.columns([3, 1], vertical_alignment="center")
        with header_l:
            st.markdown(f"## Build #{idx}")
            st.caption(f"{build.get('industry', '').replace('_',' ').title()} • beginner-friendly")
        with header_r:
            st.metric("Total", money(build.get("total_price")))

        used_pct = max(0.0, min(1.0, checks["budget_used_pct"]))
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Budget Used", f"{used_pct*100:.0f}%")
        with m2:
            st.metric("Leftover", money(checks["budget_leftover"]))
        with m3:
            st.metric("Est. Draw", f"{int(checks['est_draw_w'] or 0)} W")

        st.progress(used_pct)

        st.markdown("#### Compatibility checks")
        b1, b2, b3, b4 = st.columns(4)

        def status(col, ok: bool, title: str, ok_text="OK", bad_text="Check"):
            with col:
                if ok:
                    st.success(f"{title}: {ok_text}")
                else:
                    st.warning(f"{title}: {bad_text}")

        status(b1, checks["socket_match"], "CPU socket")
        status(b2, checks["ddr_match"], "RAM type")
        status(b3, checks["ram_slots_ok"], "RAM slots")

        psu_headroom_ok = checks["psu_ok"] and checks["psu_headroom_pct"] >= 0.15
        status(b4, psu_headroom_ok, "PSU headroom", ok_text="Good", bad_text="Tight")

        if checks["psu_ok"]:
            st.caption(
                f"PSU: {int(checks['psu_w'] or 0)}W • "
                f"Headroom: {checks['psu_headroom_pct']*100:.0f}%"
            )
        else:
            st.warning("PSU may be underpowered relative to estimated draw (check your dataset / estimates).")

        tab_overview, tab_parts, tab_details, tab_copy = st.tabs(
            ["✅ Overview", "🧩 Parts", "🔎 Details", "📋 Copy"]
        )

        with tab_overview:
            st.markdown("#### Beginner Summary (why this build makes sense)")
            for b in beginner_summary(build, checks):
                st.write(f"- {b}")

        with tab_parts:
            left, right = st.columns(2)

            with left:
                st.markdown("### Core components")
                st.caption("These determine most of your performance.")

                cpu_name = build.get("cpu", "—")
                st.markdown(f"**CPU:** {cpu_name}  \n{money(build.get('cpu_price'))}")
                st.caption("CPU matters most for productivity, simulation, and some games at high FPS.")
                part_link(
                    "CPU",
                    cpu_name,
                    [f"{build.get('cpu_cores','')} cores", f"socket {build.get('cpu_socket','')}"],
                    use=link_source,
                )

                st.divider()

                gpu_name = build.get("gpu", "—")
                st.markdown(f"**GPU:** {gpu_name}  \n{money(build.get('gpu_price'))}")
                st.caption("GPU matters most for gaming FPS, 3D work, and GPU-accelerated editing.")
                part_link("GPU", gpu_name, [f"{build.get('gpu_vram_gb','')}GB VRAM"], use=link_source)

                st.divider()

                ram_name = build.get("ram", "—")
                st.markdown(f"**RAM:** {ram_name}  \n{money(build.get('ram_price'))}")
                st.caption("More RAM helps multitasking, creation work, and engineering tools.")
                part_link(
                    "RAM",
                    ram_name,
                    [f"{build.get('ram_total_gb','')}GB", f"DDR{build.get('ram_ddr','')}"],
                    use=link_source,
                )

            with right:
                st.markdown("### Platform & power")
                st.caption("These keep things compatible and stable.")

                mb_name = build.get("motherboard", "—")
                st.markdown(f"**Motherboard:** {mb_name}  \n{money(build.get('mb_price'))}")
                st.caption("Motherboard determines CPU compatibility, RAM type, and upgrade options.")
                part_link(
                    "Motherboard",
                    mb_name,
                    [f"socket {build.get('mb_socket','')}", f"DDR{build.get('mb_ddr','')}"],
                    use=link_source,
                )

                st.divider()

                psu_name = build.get("psu", "—")
                st.markdown(f"**PSU:** {psu_name}  \n{money(build.get('psu_price'))}")
                st.caption("PSU quality/headroom matters for stability and future upgrades.")
                part_link("PSU", psu_name, [f"{build.get('psu_wattage','')}W"], use=link_source)

                st.caption(f"Estimated system draw: ~{build.get('est_draw_w','—')} W")

        with tab_details:
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

        with tab_copy:
            summary = build_summary_text(build, idx)
            st.code(summary, language="text")
            st.download_button(
                "Download summary (TXT)",
                data=summary.encode("utf-8"),
                file_name=f"build_{idx}_summary.txt",
                mime="text/plain",
                key=f"dl_summary_{idx}",
            )


# =============================
# Generate builds
# =============================
if st.button("🚀 Generate Builds", type="primary", use_container_width=True):
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
