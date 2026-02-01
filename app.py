import streamlit as st
from urllib.parse import quote_plus
from recommender import recommend_builds_from_csv_dir

st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

# Keep generating/ranking as many as you do now,
# but only display the top 5 to the user.
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
    # You can swap this for a PCPartPicker search URL if you prefer.
    # Google tends to work reliably for *any* part name quality.
    return f"https://www.google.com/search?q={quote_plus(query)}"

def pcpp_search_url(query: str) -> str:
    # Optional alternative:
    # PCPartPicker search is sometimes pickier about naming, but nicer for shopping.
    return f"https://pcpartpicker.com/search/?q={quote_plus(query)}"

def part_link(label: str, part_name: str, extras: list[str], use="google"):
    q = build_search_query(part_name, extras)
    if not q:
        st.caption("Lookup: —")
        return

    url = google_search_url(q) if use == "google" else pcpp_search_url(q)
    # Streamlit doesn't reliably open new tabs via buttons across versions,
    # so markdown links are the most compatible.
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
    # Use whatever exists. These are the main columns your build cards reference.
    candidates = ["cpu", "gpu", "ram", "motherboard", "psu"]
    return [c for c in candidates if c in df.columns]

def select_diverse_builds(df, n=5, max_overlap=1, repeat_penalty=0.35, part_cols=None):
    """
    Greedy diversification:
    - Start from best-ranked row (df already sorted by your recommender)
    - Prefer candidates that share fewer identical parts with already-selected builds
    - Also penalize frequently repeated parts across the selected set
    """
    if df is None or df.empty:
        return df

    part_cols = part_cols or get_part_cols(df)
    if not part_cols:
        return df.head(n)

    selected_idx = []
    part_counts = {}  # part string -> count among selected

    def row_parts(row):
        parts = []
        for c in part_cols:
            s = clean_str(row.get(c))
            if s:
                parts.append(s)
        return parts

    def overlap_count(parts_a, parts_b):
        # Count identical part strings overlap
        return len(set(parts_a).intersection(set(parts_b)))

    rows = df.to_dict(orient="records")

    for i, r in enumerate(rows):
        if len(selected_idx) == 0:
            selected_idx.append(i)
            for p in row_parts(r):
                part_counts[p] = part_counts.get(p, 0) + 1
            continue

        if len(selected_idx) >= n:
            break

        cand_parts = row_parts(r)

        # Overlap constraint (hard-ish rule)
        overlaps = []
        for si in selected_idx:
            overlaps.append(overlap_count(cand_parts, row_parts(rows[si])))
        worst_overlap = max(overlaps) if overlaps else 0
        if worst_overlap > max_overlap:
            continue

        # Soft preference against repeated parts
        repeat_score = sum(part_counts.get(p, 0) for p in cand_parts)
        # Lower is better. We accept greedily if it isn't too repetitive.
        # As builds fill up, we allow a bit more repetition.
        allowed = (len(selected_idx) + 1) * repeat_penalty
        if repeat_score <= allowed:
            selected_idx.append(i)
            for p in cand_parts:
                part_counts[p] = part_counts.get(p, 0) + 1
        else:
            # Still accept if we might run out later: keep this candidate as fallback.
            # We'll handle fallback fill after loop.
            continue

    # Fallback: if we didn't get enough, fill with next-best (in order)
    if len(selected_idx) < n:
        for i in range(len(rows)):
            if i in selected_idx:
                continue
            selected_idx.append(i)
            if len(selected_idx) >= n:
                break

    out = df.iloc[selected_idx].copy()
    return out

# -----------------------------
# UI controls for uniqueness + link source
# -----------------------------
with st.expander("Display options"):
    link_source = st.selectbox("Part lookup links", ["google", "pcpartpicker"], index=0)
    st.caption("Google works best with imperfect part names; PCPartPicker is nicer when names are exact.")

    make_unique = st.checkbox("Make top 5 builds more unique", value=True)
    max_overlap = st.slider(
        "Allowable overlap per build (lower = more unique)",
        min_value=0, max_value=4, value=1,
        help="This limits how many identical parts (CPU/GPU/RAM/MB/PSU) a candidate can share with an already-selected build."
    )
    repeat_penalty = st.slider(
        "Repetition tolerance (lower = fewer repeated parts)",
        min_value=0.0, max_value=2.0, value=0.35, step=0.05,
        help="Soft penalty that discourages reusing the same parts across the top 5."
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
        part_link(
            "CPU link",
            cpu_name,
            [f"{build.get('cpu_cores','')} cores", f"socket {build.get('cpu_socket','')}"],
            use=link_source
        )

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
        part_link(
            "Motherboard link",
            mb_name,
            [f"socket {build.get('mb_socket','')}", f"DDR{build.get('mb_ddr','')}"],
            use=link_source
        )

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
        # Diversify top builds if enabled
        if make_unique:
            diverse = select_diverse_builds(
                df,
                n=DISPLAY_TOP,
                max_overlap=max_overlap,
                repeat_penalty=repeat_penalty,
                part_cols=get_part_cols(df)
            )
            shown_df = diverse
        else:
            shown_df = df.head(DISPLAY_TOP)

        st.success(f"Generated {len(df)} ranked builds. Showing {len(shown_df)} build(s).")

        builds = shown_df.to_dict(orient="records")
        for i, b in enumerate(builds, start=1):
            build_card(b, i)

        # Download full ranked set (top_k), even though UI shows only top 5
        st.download_button(
            "Download ranked builds (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K}_{industry}_{int(budget)}.csv",
            mime="text/csv"
        )
