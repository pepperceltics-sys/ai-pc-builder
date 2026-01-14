import streamlit as st
from recommender import recommend_builds_from_excel

st.set_page_config(page_title="AI PC Builder", layout="wide")
st.title("AI PC Builder")
st.caption("Upload your Cleaned_data_enriched Excel file, choose an industry + budget, and generate the best compatible builds.")

# Two inputs (plus the file upload)
industry = st.selectbox("Industry", ["gaming", "office", "engineering", "content_creation"])
budget = st.number_input("Budget (USD)", min_value=300, max_value=10000, value=2000, step=50)

uploaded = st.file_uploader("Upload Cleaned_data_enriched.xlsx", type=["xlsx"])

TOP_K = 50

if st.button("Generate Builds", type="primary"):
    if uploaded is None:
        st.error("Please upload your Cleaned_data_enriched.xlsx file first.")
    else:
        with st.spinner("Generating best builds..."):
            df = recommend_builds_from_excel(
                excel_file=uploaded,
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
