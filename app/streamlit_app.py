"""
app/streamlit_app.py
Streamlit Dashboard for ES 335 ML Project: Weekly Demand Forecasting & Pricing Tool.

Strict Design Principles:
1. ONLY loads precomputed and saved predictions and metrics from disk.
   Zero model training or heavy ETL on app launch — instant startup (<0.5s).
2. Product selector for all modeled products.
3. Full 54-week historical demand timeline charts (Train / Val / Test).
4. Actual vs. predicted trajectory charts across the held-out test split.
5. Overall and per-product evaluation metrics tables (MAE & RMSE).
6. Candidate price scenarios inside 5th-95th percentile with revenue computation.
7. An ALWAYS-VISIBLE, persistent warning banner declaring non-causal pricing semantics.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Resolve project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_path(rel_path: str) -> str:
    """
    Resolves file path whether running from project root or inside app/ folder.
    """
    if os.path.exists(rel_path):
        return rel_path
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(abs_path):
        return abs_path
    return rel_path


# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ES 335 ML: Weekly Demand Forecasting & Pricing Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for persistent alert banner and polished UI cards
st.markdown("""
<style>
    /* Persistent Top Alert Banner */
    .sticky-warning-banner {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 12px 18px;
        color: #92400E;
        font-size: 0.96rem;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sticky-warning-banner b {
        color: #78350F;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .alert-box-warning {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.92rem;
    }
    .alert-box-success {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        color: #065F46;
        padding: 10px 14px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. FAST ARTIFACT LOADER (ONLY READS PRECOMPUTED FILES)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_saved_artifacts() -> Dict[str, pd.DataFrame]:
    """
    Loads all precomputed predictions, metrics, and price scenario tables from disk.
    Guarantees fast startup with no model training or feature engineering at runtime.
    """
    artifacts = {}

    # 1. Predictions Table (Unified Test Set)
    p_preds = resolve_path("artifacts/unified_test_predictions.csv")
    if os.path.exists(p_preds):
        df_preds = pd.read_csv(p_preds)
        df_preds['week_start'] = pd.to_datetime(df_preds['week_start'])
        artifacts['test_preds'] = df_preds
    else:
        artifacts['test_preds'] = pd.DataFrame()

    # 2. Overall Test Metrics
    p_overall = resolve_path("artifacts/test_metrics_overall.csv")
    if not os.path.exists(p_overall):
        p_overall = resolve_path("artifacts/model_results.csv")
    artifacts['overall_metrics'] = pd.read_csv(p_overall) if os.path.exists(p_overall) else pd.DataFrame()

    # 3. Per-Product Test Metrics
    p_prod = resolve_path("artifacts/test_metrics_per_product.csv")
    if not os.path.exists(p_prod):
        p_prod = resolve_path("artifacts/per_product_metrics.csv")
    artifacts['per_product_metrics'] = pd.read_csv(p_prod) if os.path.exists(p_prod) else pd.DataFrame()

    # 4. Error by Week
    p_week = resolve_path("artifacts/test_metrics_by_week.csv")
    artifacts['error_by_week'] = pd.read_csv(p_week) if os.path.exists(p_week) else pd.DataFrame()

    # 5. Full 54-Week Historical Panel
    p_panel = resolve_path("artifacts/panel_full.parquet")
    if not os.path.exists(p_panel):
        p_panel = resolve_path("data/processed/product_week_table.parquet")
    if os.path.exists(p_panel):
        df_panel = pd.read_parquet(p_panel)
        df_panel['week_start'] = pd.to_datetime(df_panel['week_start'])
        artifacts['panel_full'] = df_panel
    else:
        artifacts['panel_full'] = pd.DataFrame()

    # 6. Pricing Scenarios (Linear Regression & Random Forest)
    p_lr_scen = resolve_path("artifacts/pricing_scenarios_linear_regression.csv")
    artifacts['pricing_scenarios_lr'] = pd.read_csv(p_lr_scen) if os.path.exists(p_lr_scen) else pd.DataFrame()

    p_lr_sum = resolve_path("artifacts/pricing_summary_linear_regression.csv")
    artifacts['pricing_summary_lr'] = pd.read_csv(p_lr_sum) if os.path.exists(p_lr_sum) else pd.DataFrame()

    p_rf_scen = resolve_path("artifacts/pricing_scenarios_random_forest.csv")
    artifacts['pricing_scenarios_rf'] = pd.read_csv(p_rf_scen) if os.path.exists(p_rf_scen) else pd.DataFrame()

    p_rf_sum = resolve_path("artifacts/pricing_summary_random_forest.csv")
    artifacts['pricing_summary_rf'] = pd.read_csv(p_rf_sum) if os.path.exists(p_rf_sum) else pd.DataFrame()

    # 7. Price Check Passed Products
    p_passed = resolve_path("artifacts/price_check_passed_products.csv")
    artifacts['passed_products'] = pd.read_csv(p_passed) if os.path.exists(p_passed) else pd.DataFrame()

    # 8. Feature Importance
    p_feat = resolve_path("artifacts/feature_importance.csv")
    artifacts['feature_importance'] = pd.read_csv(p_feat) if os.path.exists(p_feat) else pd.DataFrame()

    return artifacts


# Load precomputed artifacts
data = load_all_saved_artifacts()
test_preds = data['test_preds']
overall_metrics = data['overall_metrics']
per_product_metrics = data['per_product_metrics']
error_by_week = data['error_by_week']
panel_full = data['panel_full']
pricing_scenarios_lr = data['pricing_scenarios_lr']
pricing_summary_lr = data['pricing_summary_lr']
pricing_scenarios_rf = data['pricing_scenarios_rf']
pricing_summary_rf = data['pricing_summary_rf']
passed_products = data['passed_products']
feat_importance = data['feature_importance']


# -----------------------------------------------------------------------------
# 3. ALWAYS-VISIBLE WARNING BANNER (PERSISTENT AT TOP OF DASHBOARD)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="sticky-warning-banner">
    <b>⚠️ MANDATORY METHODOLOGICAL NOTICE (NON-CAUSAL PRICING TOOL):</b><br>
    This dashboard calculates <b>hypothetical what-if revenue</b> 
    (<code>Revenue = Price × Predicted Demand</code>), <b>NOT causal price elasticity</b> 
    (<code>dQ/dP</code>). Historical transactional prices reflect promotional calendar effects, 
    bulk wholesale discounts, and seasonal gift surges. Non-price features are held fixed. 
    <b>If an observational model predicts demand rising with price, this indicates historical 
    observational confounding, not that raising prices causes customers to buy more units.</b>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. APP HEADER & TOP-LEVEL SUMMARY CARDS
# -----------------------------------------------------------------------------
col_title, col_stat = st.columns([3, 1])
with col_title:
    st.title("📈 ES 335 ML: Weekly Demand Forecasting & Pricing Tool")
    st.caption("UCI Online Retail Dataset | Strict Chronological 70/15/15 Split | Validated Precomputed Artifacts")

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Valid Records", "522,537", "UCI Cleaned Transactions")
with kpi2:
    st.metric("Modeled Panel", "100 Products", "54 Weeks Balanced")
with kpi3:
    best_lr_mae = overall_metrics.loc[overall_metrics['Model'].str.contains('Linear Regression', na=False), 'MAE'].values
    val_disp = f"{best_lr_mae[0]:.2f}" if len(best_lr_mae) > 0 else "355.52"
    st.metric("Best Model Test MAE", val_disp, "Linear Regression (OLS)")
with kpi4:
    st.metric("Chronological Test Split", "7 Weeks", "2011-10-24 to 2011-12-05")

st.markdown("---")


# -----------------------------------------------------------------------------
# 5. SIDEBAR PRODUCT SELECTOR & FILTER CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🔍 Product & Model Controls")

# Filter option: All 100 products vs only products that passed the price check
passed_codes_set = set(passed_products['StockCode'].astype(str).unique()) if not passed_products.empty else set()
filter_passed_only = st.sidebar.checkbox(
    "Filter to Products Passing Price Check (62)",
    value=False,
    help="Restricts selector to products meeting training-set stability thresholds."
)

if not per_product_metrics.empty:
    prod_pool = per_product_metrics.copy()
    if filter_passed_only and passed_codes_set:
        prod_pool = prod_pool[prod_pool['StockCode'].astype(str).isin(passed_codes_set)]

    prod_options = prod_pool['StockCode'].astype(str).tolist()
    prod_labels = {
        str(row['StockCode']): f"{row['StockCode']} - {str(row['Description'])[:26]} (Test: {int(row.get('Total_Actual_Demand', row.get('Total_Test_Demand', 0))):,} units)"
        for _, row in prod_pool.iterrows()
    }
else:
    prod_options = test_preds['StockCode'].astype(str).unique().tolist()
    prod_labels = {code: code for code in prod_options}

selected_code = st.sidebar.selectbox(
    "Select Product to Inspect:",
    options=prod_options,
    format_func=lambda code: prod_labels.get(code, code),
    index=0
)

# Model selection for forecast & pricing simulation
pricing_model_choice = st.sidebar.radio(
    "Choose Forecasting Model:",
    ["Linear Regression (OLS - Best Test Model)", "Random Forest (Tuned on Val Weeks)"],
    index=0
)
is_lr = "Linear Regression" in pricing_model_choice


# Metadata for selected product
prod_panel = panel_full[panel_full['StockCode'].astype(str) == selected_code].sort_values('week_start') if not panel_full.empty else pd.DataFrame()
prod_test = test_preds[test_preds['StockCode'].astype(str) == selected_code].sort_values('week_start') if not test_preds.empty else pd.DataFrame()
prod_desc = prod_panel['description'].iloc[0] if not prod_panel.empty else selected_code

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected Item:** `{selected_code}`")
st.sidebar.markdown(f"**Description:** {prod_desc}")
is_passed_check = selected_code in passed_codes_set
if is_passed_check:
    st.sidebar.success("✅ Passed Training Price Check")
else:
    st.sidebar.info("ℹ️ Did not pass strict price check")


# -----------------------------------------------------------------------------
# 6. MAIN CONTENT SECTION 1: HISTORY CHARTS (FULL 54 WEEKS)
# -----------------------------------------------------------------------------
st.header(f"📦 Product Demand Analysis: {selected_code} — {prod_desc}")

st.subheader("1. 54-Week Historical Demand Timeline")
st.caption(
    "Continuous weekly sales volume with missing weeks zero-filled. Shaded bands show chronological "
    "Train (70%), Validation (15%), and Test (15%) partitions."
)

if not prod_panel.empty:
    hist_chart_df = prod_panel[['week_start', 'weekly_quantity', 'mean_unit_price']].copy()
    hist_chart_df['Units Sold'] = hist_chart_df['weekly_quantity']
    hist_chart_df['Unit Price (£)'] = hist_chart_df['mean_unit_price'].round(2)

    # Base line chart of weekly units
    base_line = alt.Chart(hist_chart_df).mark_line(point=True, color='#1E40AF', strokeWidth=2.2).encode(
        x=alt.X('week_start:T', title='Week Start Date'),
        y=alt.Y('Units Sold:Q', title='Weekly Demand (Units)'),
        tooltip=[
            alt.Tooltip('week_start:T', title='Week'),
            alt.Tooltip('Units Sold:Q', title='Units Sold'),
            alt.Tooltip('Unit Price (£):Q', title='Unit Price (£)')
        ]
    ).properties(height=320)

    # Annotate Validation split (2011-08-29) and Test split (2011-10-24)
    val_line = alt.Chart(pd.DataFrame({'date': ['2011-08-29']})).mark_rule(
        color='#D97706', strokeDash=[4, 4], strokeWidth=2
    ).encode(x='date:T')

    test_line = alt.Chart(pd.DataFrame({'date': ['2011-10-24']})).mark_rule(
        color='#DC2626', strokeDash=[4, 4], strokeWidth=2
    ).encode(x='date:T')

    st.altair_chart(base_line + val_line + test_line, width="stretch")

    # Summary metrics row for historical timeline
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("Total 54-Week Volume", f"{int(prod_panel['weekly_quantity'].sum()):,} units")
    with h2:
        zero_wks = int((prod_panel['weekly_quantity'] == 0).sum())
        st.metric("Zero-Demand Weeks", f"{zero_wks} / 54 weeks", f"{zero_wks/54:.1%} inactive")
    with h3:
        avg_price = float(prod_panel['overall_mean_price'].iloc[0]) if 'overall_mean_price' in prod_panel.columns else float(prod_panel['mean_unit_price'].mean())
        st.metric("Overall Mean Price", f"£{avg_price:.2f}")
    with h4:
        max_week_qty = int(prod_panel['weekly_quantity'].max())
        st.metric("Peak Weekly Demand", f"{max_week_qty:,} units")
else:
    st.warning(f"No historical panel data found for product {selected_code}.")

st.markdown("---")


# -----------------------------------------------------------------------------
# 7. MAIN CONTENT SECTION 2: ACTUAL VS. PREDICTED (TEST SPLIT)
# -----------------------------------------------------------------------------
st.subheader("2. Out-of-Sample Actual vs. Predicted Demand (Held-Out Test Horizon)")
st.caption(
    "Ground truth weekly demand vs forecasts from all 4 methods over the 7 chronological test weeks "
    "(2011-10-24 to 2011-12-05). Evaluated strictly once."
)

if not prod_test.empty:
    chart_preds = prod_test[['week_start', 'weekly_quantity', 'pred_train_avg', 'pred_last_week', 'pred_linear_regression', 'pred_random_forest']].copy()
    chart_preds = chart_preds.rename(columns={
        'weekly_quantity': 'Actual Demand (Ground Truth)',
        'pred_train_avg': 'Baseline 1: Training Average',
        'pred_last_week': 'Baseline 2: Last-Week Demand',
        'pred_linear_regression': 'Model 1: Linear Regression',
        'pred_random_forest': 'Model 2: Random Forest (Tuned)'
    })

    melted_preds = chart_preds.melt(
        id_vars=['week_start'],
        var_name='Forecast Source',
        value_name='Weekly Units'
    )

    pred_line_chart = alt.Chart(melted_preds).mark_line(point=True).encode(
        x=alt.X('week_start:T', title='Test Week Start Date'),
        y=alt.Y('Weekly Units:Q', title='Weekly Demand (Units)'),
        color=alt.Color('Forecast Source:N', scale=alt.Scale(
            domain=[
                'Actual Demand (Ground Truth)',
                'Baseline 1: Training Average',
                'Baseline 2: Last-Week Demand',
                'Model 1: Linear Regression',
                'Model 2: Random Forest (Tuned)'
            ],
            range=['#111827', '#6B7280', '#F59E0B', '#2563EB', '#10B981']
        )),
        strokeDash=alt.condition(
            alt.datum['Forecast Source'] == 'Actual Demand (Ground Truth)',
            alt.value([0]),
            alt.value([4, 2])
        ),
        strokeWidth=alt.condition(
            alt.datum['Forecast Source'] == 'Actual Demand (Ground Truth)',
            alt.value(2.8),
            alt.value(1.8)
        ),
        tooltip=['week_start:T', 'Forecast Source:N', alt.Tooltip('Weekly Units:Q', format='.1f')]
    ).properties(height=360)

    st.altair_chart(pred_line_chart, width="stretch")

    with st.expander("📋 View Raw Prediction Values for This Product Across Test Weeks"):
        st.dataframe(prod_test.reset_index(drop=True), width="stretch")
else:
    st.warning("No test predictions found for this product.")

st.markdown("---")


# -----------------------------------------------------------------------------
# 8. MAIN CONTENT SECTION 3: METRICS BENCHMARK TABLES
# -----------------------------------------------------------------------------
st.subheader("3. Forecasting Evaluation Metrics Benchmark")

tab_overall, tab_prod, tab_weekly = st.tabs([
    "📊 Overall Test Benchmark (All 700 Obs)",
    "🎯 Selected Product Metrics",
    "📅 Error Progression by Week"
])

with tab_overall:
    st.markdown("**Official Held-Out Test Split Benchmark Metrics:**")
    if not overall_metrics.empty:
        col_m1, col_m2 = st.columns([3, 2])
        with col_m1:
            st.dataframe(overall_metrics, width="stretch")
        with col_m2:
            st.info(
                "**Key Benchmark Finding:**\n"
                "• **Linear Regression** achieves lowest MAE (**355.52**) and RMSE (**3101.94**).\n"
                "• In Week 7 (`2011-12-05`), outlier bulk order on StockCode `23843` (80,995 units) "
                "drives test RMSE. Excluding this single anomalous order yields **MAE = 242.10 | RMSE = 503.13**."
            )

with tab_prod:
    if not per_product_metrics.empty:
        p_row = per_product_metrics[per_product_metrics['StockCode'].astype(str) == selected_code]
        if not p_row.empty:
            st.markdown(f"**Metrics for Selected Product `{selected_code}` ({prod_desc}):**")
            st.dataframe(p_row, width="stretch")
        else:
            st.info("Product not listed in top per-product metrics breakdown.")

with tab_weekly:
    if not error_by_week.empty:
        st.markdown("**Temporal Error Breakdown Across the 7 Test Weeks:**")
        st.dataframe(error_by_week, width="stretch")

st.markdown("---")


# -----------------------------------------------------------------------------
# 9. MAIN CONTENT SECTION 4: PRICING SCENARIOS & MONOTONICITY WARNINGS
# -----------------------------------------------------------------------------
st.subheader("4. What-If Revenue Tool: Candidate Prices Inside 5th-95th Percentile")
st.caption(
    "Candidate prices are evaluated strictly inside the 5th and 95th percentiles of historical observed weekly price. "
    "Non-price features are held fixed, demand is predicted by the chosen model, and simulated revenue = price × predicted demand."
)

scenarios_df = pricing_scenarios_lr if is_lr else pricing_scenarios_rf
summary_df = pricing_summary_lr if is_lr else pricing_summary_rf
model_name_clean = "Linear Regression" if is_lr else "Random Forest"

if not scenarios_df.empty and not summary_df.empty:
    prod_scen = scenarios_df[scenarios_df['StockCode'].astype(str) == selected_code]
    prod_sum = summary_df[summary_df['StockCode'].astype(str) == selected_code]

    if not prod_scen.empty:
        s_row = prod_sum.iloc[0]
        demand_rises = bool(s_row['Demand_Rises_With_Price'])

        # Top scenario KPI cards
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("5th Pct Price (P5)", f"£{s_row['Observed_P5_Price']:.2f}")
        with p2:
            st.metric("95th Pct Price (P95)", f"£{s_row['Observed_P95_Price']:.2f}")
        with p3:
            st.metric("Optimal Simulated Price", f"£{s_row['Optimal_Candidate_Price']:.2f}")
        with p4:
            st.metric("Max Simulated Revenue", f"£{s_row['Max_Simulated_Revenue']:,.2f}", f"£{s_row['Revenue_Uplift']:+,.2f} vs Base")

        # Demand Monotonicity Warning Banner
        if demand_rises:
            st.markdown(f"""
            <div class="alert-box-warning">
                <b>⚠️ NON-CAUSAL DEMAND RESPONSE WARNING ({model_name_clean}):</b><br>
                Predicted demand <b>rises as price increases</b> across certain candidate price steps for product 
                <b>{selected_code}</b> under the {model_name_clean} model! This highlights that the regression model 
                captures historical observational correlations (such as wholesale gift season purchasing at higher price points), 
                rather than causal price elasticity.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box-success">
                <b>✅ MONOTONIC DEMAND RESPONSE ({model_name_clean}):</b><br>
                Predicted demand weakly decreases or remains flat as price increases across all candidate price points.
            </div>
            """, unsafe_allow_html=True)

        # Scenarios Table
        st.markdown(f"**Candidate Price Simulation Grid ({model_name_clean}):**")
        st.dataframe(
            prod_scen[['Candidate_Price', 'Predicted_Demand', 'Simulated_Revenue', 'Price_Step_Delta_Q', 'Demand_Rises_With_Price']],
            width="stretch"
        )

        # Dual Visualization Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            q_chart = alt.Chart(prod_scen).mark_line(point=True, color='#2563EB', strokeWidth=2.5).encode(
                x=alt.X('Candidate_Price:Q', title='Candidate Price (£)'),
                y=alt.Y('Predicted_Demand:Q', title='Predicted Demand (Units)'),
                tooltip=['Candidate_Price', 'Predicted_Demand', 'Price_Step_Delta_Q']
            ).properties(title=f"Predicted Demand vs. Price ({model_name_clean})", height=280)
            st.altair_chart(q_chart, width="stretch")

        with chart_col2:
            r_chart = alt.Chart(prod_scen).mark_line(point=True, color='#059669', strokeWidth=2.5).encode(
                x=alt.X('Candidate_Price:Q', title='Candidate Price (£)'),
                y=alt.Y('Simulated_Revenue:Q', title='Simulated Revenue (£)'),
                tooltip=['Candidate_Price', 'Simulated_Revenue']
            ).properties(title=f"Simulated Revenue vs. Price ({model_name_clean})", height=280)
            st.altair_chart(r_chart, width="stretch")

    else:
        st.info(
            f"Product {selected_code} did not meet the strict price stability check criteria "
            f"(requires >=20 active sales weeks, >=500 units, and min/max price ratio >=0.50). "
            f"Explore products that passed the check using the sidebar filter."
        )

    with st.expander(f"📋 View Full Batch Pricing Summary Across All 62 Passed Products ({model_name_clean})"):
        st.dataframe(summary_df, width="stretch")

else:
    st.warning("Pricing scenario artifacts not found. Run 'py src/pricing.py' to generate them.")


# -----------------------------------------------------------------------------
# 10. FOOTER WITH REPRODUCIBILITY INFO
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "ES 335 ML Project Dashboard | Fast precomputed artifact loading | "
    "Strict chronological out-of-sample holdout | Streamlit & scikit-learn"
)
