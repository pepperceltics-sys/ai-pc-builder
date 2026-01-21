import streamlit as st
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

def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"

def show_if_present(label: str, value):
    if value is None:
        return
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return
    st.write(f"**{label}:** {value}")

def build_search_query(part_name: str, extras: list[str]) -> str:
    base = (part_name or "").strip()
    extras_clean = [e.strip() for e in extras if e and str(e).strip() not in ("", "nan", "None")]
    if not base and not extras_clean:
        return "—"
    return " ".join([base] + extras_clean).strip()

def build_summary_text(build: dict, idx: int) -> str:
    """
    One-line-per-part summary users can copy/paste into Reddit/Discord/etc.
    Uses whatever the dataset provides (so it stays robust).
    """
    lines = []
    lines.append(f"Build #{idx} — Total: {money(build.get('total_price'))} — Industry: {str(build.get('industry','')).capitalize()}")
    lines.append(f"CPU: {build.get('cpu','—')} ({build.get('cpu_cores','—')} cores, socket {build.get('cpu_socket','—')}) — {money(build.get('cpu_price'))}")
    lines.append(f"GPU: {build.get('gpu','—')} ({build.get('gpu_vram_gb','—')}GB VRAM) — {money(build.get('gpu_price'))}")
    lines.append(f"RAM: {build.get('ram','—')} ({build.get('ram_total_gb','—')}GB, DDR{build.get('ram_ddr','—')}) — {money(build.get('ram_price'))}")
    lines.append(f"Motherboard: {build.get('motherboard','—')} (socket {build.get('mb_socket','—')}, DDR{build.get('mb_ddr','—')}) — {money(build.get('mb_price'))}")
    lines.append(f"PSU: {build.get('psu','—')} ({build.get('psu_wattage','—')}W) — {money(build.get('psu_price'))}")
    lines.append(f"Est. draw: ~{build.get('est_draw_w','—')}W")
    return "\n".join(lines)

def build_card(build: dict, idx: int):
    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"Build #{idx}")
        st.caption(f"{build.get('industry', '').capitalize()} build")
    with right:
        st.metric("Total", money(build.get("total_price")))

    parts_left, parts_right = st.columns([2, 2])

    # ----- Core components -----
    with parts_left:
        st.markdown("**Core components**")

        cpu_name = build.get("cpu", "—")
        cpu_query = build_search_query(cpu_name, [
            f"{build.get('cpu_cores','')} cores" if build.get("cpu_cores") else "",
            f"socket {build.get('cpu_socket','')}" if build.get("cpu_socket") else "",
        ])
        st.write(f"**CPU (Model):** {cpu_name} — {money(build.get('cpu_price'))}")
        st.caption(f"Search: `{cpu_query}`")

        gpu_name = build.get("gpu", "—")
        gpu_query = build_search_query(gpu_name, [
            f"{build.get('gpu_vram_gb','')}GB VRAM" if build.get("gpu_vram_gb") else "",
        ])
        st.write(f"**GPU (Model):** {gpu_name} — {money(build.get('gpu_price'))}")
        st.caption(f"Search: `{gpu_query}`")

        ram_name = build.get("ram", "—")
        ram_query = build_search_query(ram_name, [
            f"{build.get('ram_total_gb','')}GB" if build.get("ram_total_gb") else "",
            f"DDR{build.get('ram_ddr','')}" if build.get("ram_ddr") else "",
        ])
        st.write(f"**RAM (Model):** {ram_name} — {money(build.get('ram_price'))}")
        st.caption(f"Search: `{ram_query}`")

    # ----- Platform & power -----
    with parts_right:
        st.markdown("**Platform & power**")

        mb_name = build.get("motherboard", "—")
        mb_query = build_search_query(mb_name, [
            f"socket {build.get('mb_socket','')}" if build.get("mb_socket") else "",
            f"DDR{build.get('mb_ddr','')}" if build.get("mb_ddr") else "",
        ])
        st.write(f"**Motherboard (Model):** {mb_name} — {money(build.get('mb_price'))}")
        st.caption(f"Search: `{mb_query}`")

        psu_name = build.get("psu", "—")
        psu_query = build_search_query(psu_name, [
            f"{build.get('psu_wattage','')}W" if build.get("psu_wattage") else "",
        ])
        st.write(f"**PSU (Model):** {psu_name} — {money(build.get('psu_price'))}")
        st.caption(f"Search: `{psu_query}`")

        st.caption(f"Estimated system draw: ~{build.get('est_draw_w','—')} W")

    # ----- Copy-friendly summary -----
    with st.expander("Copy build summary"):
        summary = build_summary_text(build, idx)
        # Streamlit's code blocks typically show a copy button in the UI.
        st.code(summary, language="text")
        # Optional: also provide a download for the summary text
        st.download_button(
            "Download summary (TXT)",
            data=summary.encode("utf-8"),
            file_name=f"build_{idx}_summary.txt",
            mime="text/plain",
            key=f"dl_summary_{idx}",
        )

    # ----- More lookup-friendly details -----
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

if st.button("Generate Builds", type="primary"):
    with st.spinner("Generating best builds..."):
        df = recommend_builds_from_csv_dir(
            data_dir=DATA_DIR,
            industry=industry,
            total_budget=float(budget),
            top_k=TOP_K
        )

    if df.empty:
        st.warning("No compatible builds found under these constraints. Try increasing your budget.")
    else:
        st.success(f"Generated {len(df)} ranked builds. Showing top {min(DISPLAY_TOP, len(df))}.")

        # Show only top 5 as build cards
        top = df.head(DISPLAY_TOP).to_dict(orient="records")
        for i, b in enumerate(top, start=1):
            build_card(b, i)

        # Download full ranked set (top_k), even though UI shows only top 5
        st.download_button(
            "Download ranked builds (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K}_{industry}_{int(budget)}.csv",
            mime="text/csv"
        )
