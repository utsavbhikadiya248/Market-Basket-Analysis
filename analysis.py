
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

DATA_PATH = os.path.join("data", "Online Retail.xlsx")
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("=" * 60)
print("MARKET BASKET ANALYSIS FOR REVENUE OPTIMIZATION")
print("=" * 60)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Put Online Retail.xlsx inside the data folder."
    )

df = pd.read_excel(DATA_PATH)

print("\n1. ORIGINAL DATASET")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

original_rows = len(df)

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Remove records with missing transaction or product details
df = df.dropna(subset=[
    "InvoiceNo",
    "Description",
    "Quantity",
    "UnitPrice"
])

# Remove cancelled invoices
df["InvoiceNo"] = df["InvoiceNo"].astype(str)
df = df[~df["InvoiceNo"].str.startswith("C")]

# Keep valid purchases only
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Clean product descriptions
df["Description"] = df["Description"].astype(str).str.strip()
df = df[df["Description"] != ""]

# Calculate revenue per line item
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Use one country to keep the first analysis manageable
df = df[df["Country"] == "United Kingdom"].copy()

if df.empty:
    raise ValueError(
        "No UK records found. Check the Country column in the dataset."
    )

cleaned_rows = len(df)

print("\n2. DATA CLEANING")
print("Original rows:", original_rows)
print("Rows after cleaning and UK filter:", cleaned_rows)
print("Rows removed:", original_rows - cleaned_rows)
print("Country selected: United Kingdom")

df.to_csv(
    os.path.join(OUTPUT_DIR, "cleaned_data.csv"),
    index=False
)

# --------------------------------------------------
# 4. SALES AND PRODUCT ANALYSIS
# --------------------------------------------------

total_revenue = df["Revenue"].sum()
total_transactions = df["InvoiceNo"].nunique()
total_products = df["Description"].nunique()
average_transaction_value = total_revenue / total_transactions

print("\n3. BUSINESS SUMMARY")
print("Total revenue: £{:.2f}".format(total_revenue))
print("Unique transactions:", total_transactions)
print("Unique products:", total_products)
print(
    "Average transaction value: £{:.2f}".format(
        average_transaction_value
    )
)

summary = pd.DataFrame({
    "Metric": [
        "Country",
        "Rows after cleaning",
        "Unique transactions",
        "Unique products",
        "Total revenue",
        "Average transaction value"
    ],
    "Value": [
        "United Kingdom",
        cleaned_rows,
        total_transactions,
        total_products,
        round(total_revenue, 2),
        round(average_transaction_value, 2)
    ]
})

summary.to_csv(
    os.path.join(OUTPUT_DIR, "business_summary.csv"),
    index=False
)

# Top products by revenue
top_products = (
    df.groupby("Description")
    .agg(
        Total_Quantity=("Quantity", "sum"),
        Total_Revenue=("Revenue", "sum"),
        Invoice_Count=("InvoiceNo", "nunique")
    )
    .sort_values("Total_Revenue", ascending=False)
    .head(15)
    .reset_index()
)

print("\n4. TOP 15 PRODUCTS BY REVENUE")
print(top_products.to_string(index=False))

top_products.to_csv(
    os.path.join(OUTPUT_DIR, "top_products.csv"),
    index=False
)

# Save a chart of top products
plt.figure(figsize=(12, 7))
chart_data = top_products.sort_values("Total_Revenue")

plt.barh(
    chart_data["Description"],
    chart_data["Total_Revenue"]
)

plt.title("Top 15 Products by Revenue")
plt.xlabel("Revenue (£)")
plt.ylabel("Product")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "top_products.png"),
    dpi=150
)
plt.close()

# --------------------------------------------------
# 5. PREPARE TRANSACTION BASKETS
# --------------------------------------------------

# Keep the 40 most frequently purchased products.
# This limits the number of combinations examined.
product_invoice_counts = (
    df.groupby("Description")["InvoiceNo"].nunique()
    .sort_values(ascending=False)
)

top_40_products = product_invoice_counts.head(40).index

basket_data = df[
    df["Description"].isin(top_40_products)
].copy()

# Each row represents an invoice; each column represents a product.
basket = basket_data.pivot_table(
    index="InvoiceNo",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Apriori needs True/False values indicating product presence.
basket = basket > 0

print("\n5. TRANSACTION BASKET")
print("Number of transactions:", basket.shape[0])
print("Number of products analyzed:", basket.shape[1])

# --------------------------------------------------
# 6. APPLY APRIORI ALGORITHM
# --------------------------------------------------

print("\n6. APPLYING APRIORI ALGORITHM...")

frequent_itemsets = apriori(
    basket,
    min_support=0.01,
    use_colnames=True,
    max_len=3
)

if frequent_itemsets.empty:
    print(
        "No frequent itemsets found. Try lowering min_support to 0.005."
    )
    frequent_itemsets.to_csv(
        os.path.join(OUTPUT_DIR, "frequent_itemsets.csv"),
        index=False
    )
    raise ValueError("No frequent itemsets found. Analysis stopped.")

frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
    lambda items: ", ".join(sorted(items))
)

frequent_itemsets = frequent_itemsets.sort_values(
    "support", ascending=False
)

print("\nFrequent itemsets found:", len(frequent_itemsets))
print(frequent_itemsets.head(15).to_string(index=False))

frequent_itemsets.to_csv(
    os.path.join(OUTPUT_DIR, "frequent_itemsets.csv"),
    index=False
)

# Re-create itemsets with frozensets for association rule mining.
# Use the original Apriori result before changing its itemsets column.
frequent_itemsets_for_rules = apriori(
    basket,
    min_support=0.01,
    use_colnames=True,
    max_len=3
)

# --------------------------------------------------
# 7. GENERATE ASSOCIATION RULES
# --------------------------------------------------

rules = association_rules(
    frequent_itemsets_for_rules,
    metric="confidence",
    min_threshold=0.30
)

if rules.empty:
    print("\nNo association rules found at confidence 0.30.")
    rules_output = pd.DataFrame(columns=[
        "Antecedents", "Consequents", "Support",
        "Confidence", "Lift"
    ])
else:
    rules = rules[rules["lift"] > 1.0].copy()
    rules = rules.sort_values(
    ["lift", "confidence"],
    ascending=False
    )


    rules_output = rules.copy()
    rules_output["Antecedents"] = rules_output[
        "antecedents"
    ].apply(lambda items: ", ".join(sorted(items)))

    rules_output["Consequents"] = rules_output[
        "consequents"
    ].apply(lambda items: ", ".join(sorted(items)))

    rules_output = rules_output[[
        "Antecedents",
        "Consequents",
        "support",
        "confidence",
        "lift"
    ]]

    rules_output.columns = [
        "Antecedents",
        "Consequents",
        "Support",
        "Confidence",
        "Lift"
    ]

print("\n7. ASSOCIATION RULES")
print("Rules found:", len(rules_output))

if not rules_output.empty:
    print(rules_output.head(20).to_string(index=False))

rules_output.to_csv(
    os.path.join(OUTPUT_DIR, "association_rules.csv"),
    index=False
)

# --------------------------------------------------
# 8. REVENUE OPTIMIZATION RECOMMENDATIONS
# --------------------------------------------------

print("\n8. REVENUE OPTIMIZATION RECOMMENDATIONS")

if not rules_output.empty:
    recommendations = rules_output.head(10).copy()

    recommendations["Business_Recommendation"] = (
        "Consider displaying the consequent product as a "
        "cross-sell suggestion when the antecedent is purchased."
    )

    print(recommendations.to_string(index=False))

    recommendations.to_csv(
        os.path.join(OUTPUT_DIR, "recommendations.csv"),
        index=False
    )
else:
    print(
        "No rules passed the selected thresholds. "
        "Try min_support=0.005 or min_threshold=0.20."
    )

# --------------------------------------------------
# 9. FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Check the outputs folder for CSV files and the chart.")
print("Remember: association does not prove that a bundle")
print("will cause an increase in revenue. Test recommendations")
print("with a controlled promotion before claiming uplift.")
