
import os
import subprocess
import sys
import pandas as pd
import streamlit as st

OUTPUT_DIR = "outputs"

st.set_page_config(
    page_title="Market Basket Analysis",
    page_icon="",
    layout="wide"
)

st.title("Market Basket Analysis for Revenue Optimization")
st.write(
    "Analyze retail transactions, discover products frequently "
    "purchased together, and identify cross-selling opportunities."
)

st.sidebar.header("Project Controls")

if st.sidebar.button("Run Analysis"):
    with st.spinner("Running Python analysis..."):
        result = subprocess.run(
            [sys.executable, "analysis.py"],
            capture_output=True,
            text=True
        )

    if result.returncode == 0:
        st.success("Analysis completed successfully.")
        st.code(result.stdout)
    else:
        st.error("Analysis failed. Check the error below.")
        st.code(result.stderr)

st.subheader("Project Overview")

summary_path = os.path.join(OUTPUT_DIR, "business_summary.csv")
products_path = os.path.join(OUTPUT_DIR, "top_products.csv")
itemsets_path = os.path.join(OUTPUT_DIR, "frequent_itemsets.csv")
rules_path = os.path.join(OUTPUT_DIR, "association_rules.csv")
recommendations_path = os.path.join(OUTPUT_DIR, "recommendations.csv")
chart_path = os.path.join(OUTPUT_DIR, "top_products.png")

if os.path.exists(summary_path):
    summary = pd.read_csv(summary_path)

    metrics = dict(zip(summary["Metric"], summary["Value"]))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Revenue",
            "£{:,.2f}".format(float(metrics.get("Total revenue", 0)))
        )

    with col2:
        st.metric(
            "Unique Transactions",
            "{:,.0f}".format(
                float(metrics.get("Unique transactions", 0))
            )
        )

    with col3:
        st.metric(
            "Unique Products",
            "{:,.0f}".format(
                float(metrics.get("Unique products", 0))
            )
        )

    st.subheader("Business Summary")
    st.dataframe(summary, use_container_width=True)

    if os.path.exists(chart_path):
        st.subheader("Top Products by Revenue")
        st.image(chart_path, use_container_width=True)

    if os.path.exists(products_path):
        st.subheader("Top Products")
        st.dataframe(
            pd.read_csv(products_path),
            use_container_width=True
        )

    if os.path.exists(itemsets_path):
        st.subheader("Frequent Itemsets")
        st.dataframe(
            pd.read_csv(itemsets_path),
            use_container_width=True
        )

    if os.path.exists(rules_path):
        st.subheader("Association Rules")
        st.caption(
            "Support measures frequency, confidence measures "
            "rule reliability, and lift compares the association "
            "with what would be expected under independence."
        )
        st.dataframe(
            pd.read_csv(rules_path),
            use_container_width=True
        )

    if os.path.exists(recommendations_path):
        st.subheader("Revenue Optimization Recommendations")
        st.dataframe(
            pd.read_csv(recommendations_path),
            use_container_width=True
        )

    for filename in [
        "business_summary.csv",
        "top_products.csv",
        "frequent_itemsets.csv",
        "association_rules.csv",
        "recommendations.csv"
    ]:
        filepath = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(filepath):
            with open(filepath, "rb") as file:
                st.download_button(
                    label="Download " + filename,
                    data=file.read(),
                    file_name=filename,
                    mime="text/csv",
                    key=filename
                )
else:
    st.info(
        "No results found yet. Click 'Run Analysis' in the sidebar "
        "or run 'python analysis.py' in the terminal first."
    )

st.divider()
st.subheader("How This Project Helps Businesses")
st.markdown(
    """
    - **Product bundling:** identify products that often appear together.
    - **Cross-selling:** suggest related products at checkout.
    - **Promotions:** identify combinations worth testing in offers.
    - **Inventory planning:** understand which products are commonly
      purchased together.
    """
)

st.warning(
    "Association rules show patterns in historical purchases. "
    "They do not prove that a recommendation will increase revenue. "
    "Actual uplift should be measured using a controlled test."
)
