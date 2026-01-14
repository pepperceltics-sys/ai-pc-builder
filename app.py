import streamlit as st
from recommender import recommend_builds_from_csv_dir

st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Select an industry + budget. Builds are generated from CSVs stored in /data.")

industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

TOP_K = 50
DATA_DIR = "data"

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
        st.success(f"Found {len(df)} builds (showing top {min(50, len(df))}).")
        st.dataframe(df.head(50), use_container_width=True)

        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"top_{TOP_K}_{industry}_{int(budget)}.csv",
            mime="text/csv"
        )
