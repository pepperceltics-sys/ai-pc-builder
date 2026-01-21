import streamlit as st
from recommender import recommend_builds_from_csv_dir

st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

# Keep generating as many as you do now,
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

def build_card(build, idx: int):
    # Header row: Build rank + total price
    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"Build #{idx}")
        st.caption(f"{build.get('industry', '').capitalize()} build")
    with right:
        st.metric("Total", money(build.get("total_price")))

    # Parts + prices
    parts_left, parts_right = st.columns([2, 2])

    with parts_left:
        st.markdown("**Core components**")
        st.write(f"**CPU:** {build.get('cpu','—')} ({build.get('cpu_cores','—')} cores) — {money(build.get('cpu_price'))}")
        st.write(f"**GPU:** {build.get('gpu','—')} ({build.get('gpu_vram_gb','—')} GB VRAM) — {money(build.get('gpu_price'))}")
        st.write(f"**RAM:** {build.get('ram','—')} ({build.get('ram_total_gb','—')} GB, {build.get('ram_ddr','—')}) — {money(build.get('ram_price'))}")

    with parts_right:
        st.markdown("**Platform & power**")
        st.write(f"**Motherboard:** {build.get('motherboard','—')} ({build.get('mb_ddr','—')}) — {money(build.get('mb_price'))}")
        st.write(f"**PSU:** {build.get('psu','—')} ({build.get('psu_wattage','—')} W) — {money(build.get('psu_price'))}")
        st.caption(f"Estimated system draw: ~{build.get('est_draw_w','—')} W")

    # Optional details without exposing scores to the main UI
    with st.expander("Details (sockets / compatibility)"):
        c1, c2, c3 = st.columns(3)
        c1.write(f"**CPU socket:** {build.get('cpu_socket','—')}")
        c2.write(f"**MB socket:** {build.get('mb_socket','—')}")
        c3.write(f"**RAM type:** {build.get('mb_ddr','—')}")

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

        # Build cards (top 5)
        top = df.head(DISPLAY_TOP).to_dict(orient="records")
        for i, b in enumerate(top, start=1):
            build_card(b, i)

        # Download: keep it useful (top_k builds), even though UI shows only top 5
        st.download_button(
            "Download ranked builds (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K}_{industry}_{int(budget)}.csv",
            mime="text/csv"
        )
