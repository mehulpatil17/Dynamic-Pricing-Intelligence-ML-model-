"""
src/pricing.py
Module implementing the What-If Revenue Tool for products passing the price check.

Workflow:
1. Filters products that passed the training-set price check.
2. Extracts candidate selling prices strictly within the 5th to 95th percentile
   of observed weekly prices for each product.
3. Holds all non-price features fixed (lag_1, lag_2, lag_4, rolling_mean_4,
   rolling_std_4, week_of_year) at their reference values.
4. Predicts demand quantity using the chosen model across the candidate prices.
5. Computes hypothetical revenue:
       Revenue = Price * Predicted Quantity
6. Evaluates demand monotonicity and issues explicit warnings if predicted
   demand rises with price (a non-causal observational regression artifact).
"""

import os
import sys
import warnings
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Ensure parent directory is in sys.path when running script directly
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_training_split(
    df: pd.DataFrame,
    test_weeks_count: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Partitions the cleaned dataset into training and testing sets chronologically.
    """
    if 'Week_Start' not in df.columns:
        df['Week_Start'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('W-SUN').dt.start_time

    all_weeks = sorted(df['Week_Start'].unique())
    split_date = all_weeks[-test_weeks_count]

    train_df = df[df['Week_Start'] < split_date].copy().reset_index(drop=True)
    test_df = df[df['Week_Start'] >= split_date].copy().reset_index(drop=True)

    return train_df, test_df, split_date


def plot_weekly_total_units(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots the total units sold per week over the entire dataset timeline.
    """
    weekly_totals = df.groupby('Week_Start')['Quantity'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly_totals['Week_Start'], weekly_totals['Quantity'], color='#1D4ED8', marker='o', linewidth=2.2, markersize=4)
    ax.set_title("Weekly Total Units Sold (UCI Online Retail)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Week Start Date", fontsize=11)
    ax.set_ylabel("Total Units Sold", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.axvspan(pd.Timestamp('2011-10-01'), pd.Timestamp('2011-12-09'), color='#FEF3C7', alpha=0.5, label='Q4 Peak Retail Horizon')
    ax.legend(loc='upper left', frameon=True)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_top_products(
    df: pd.DataFrame,
    top_n: int = 30,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots a horizontal bar chart of the top N products by total volume.
    """
    prod_summary = df.groupby('StockCode').agg(
        total_units=('Quantity', 'sum'),
        description=('Description', 'first')
    ).reset_index()

    top_df = prod_summary.nlargest(top_n, 'total_units').sort_values(by='total_units', ascending=True)

    labels = [
        f"{row['StockCode']} - {str(row['description'])[:24]}"
        for _, row in top_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(range(len(top_df)), top_df['total_units'] / 1000.0, color='#059669', edgecolor='#047857')
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Total Units Sold (in Thousands)", fontsize=11)
    ax.set_title(f"Top {top_n} Products by Total Volume Sold", fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.1f}k",
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=8, color='#1F2937')

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_price_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots transaction-level and product-level price distributions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    prices_under_20 = df[df['UnitPrice'] <= 20]['UnitPrice']
    ax1.hist(prices_under_20, bins=40, color='#6366F1', edgecolor='#4338CA', alpha=0.85)
    ax1.axvline(df['UnitPrice'].median(), color='#DC2626', linestyle='--', linewidth=2,
                label=f"Median: £{df['UnitPrice'].median():.2f}")
    ax1.axvline(df['UnitPrice'].mean(), color='#F59E0B', linestyle=':', linewidth=2,
                label=f"Mean: £{df['UnitPrice'].mean():.2f}")
    ax1.set_title("Transaction UnitPrice Distribution (£0 - £20)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Unit Selling Price (£)", fontsize=10)
    ax1.set_ylabel("Transaction Count", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()

    prod_prices = df.groupby('StockCode')['UnitPrice'].mean()
    log_prices = np.log10(prod_prices[prod_prices > 0])
    ax2.hist(log_prices, bins=40, color='#8B5CF6', edgecolor='#6D28D9', alpha=0.85)
    ax2.set_title("Product Mean UnitPrice Distribution (Log10 Scale)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Log10(Mean UnitPrice)", fontsize=10)
    ax2.set_ylabel("Product Count", fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def compute_training_price_metrics(
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes per-product price stability and sales frequency strictly on training weeks.
    
    Metrics computed per product:
    1. Weeks_With_Sales: distinct calendar weeks with >= 1 unit sold.
    2. Distinct_Weekly_Prices: number of unique weekly price points observed.
    3. Weeks_Per_Price: ratio of active weeks to distinct prices (average weeks per price level).
    4. Min_Price, Max_Price: lowest and highest observed weekly average prices.
    5. Min_Max_Price_Ratio: Min_Price / Max_Price (measures relative price stability).
    """
    weekly_prod = train_df.groupby(['StockCode', 'Week_Start']).agg(
        weekly_units=('Quantity', 'sum'),
        weekly_price=('UnitPrice', 'mean'),
        description=('Description', 'first')
    ).reset_index()

    records = []
    for stock_code, grp in weekly_prod.groupby('StockCode'):
        weeks_with_sales = int(grp['Week_Start'].nunique())
        rounded_prices = grp['weekly_price'].round(2)
        distinct_prices = int(rounded_prices.nunique())
        min_p = float(rounded_prices.min())
        max_p = float(rounded_prices.max())

        min_max_ratio = float(min_p / max_p) if max_p > 0 else 0.0
        weeks_per_price = float(weeks_with_sales / distinct_prices) if distinct_prices > 0 else 0.0

        records.append({
            'StockCode': stock_code,
            'Description': grp['description'].iloc[0],
            'Total_Units_Train': int(grp['weekly_units'].sum()),
            'Weeks_With_Sales': weeks_with_sales,
            'Distinct_Weekly_Prices': distinct_prices,
            'Weeks_Per_Price': round(weeks_per_price, 2),
            'Min_Price': round(min_p, 2),
            'Max_Price': round(max_p, 2),
            'Min_Max_Price_Ratio': round(min_max_ratio, 3)
        })

    price_df = pd.DataFrame(records)
    return price_df.sort_values(by='Total_Units_Train', ascending=False).reset_index(drop=True)


def filter_products_by_thresholds(
    price_stats_df: pd.DataFrame,
    min_weeks_with_sales: int = 20,
    min_total_units: int = 500,
    min_price_ratio: float = 0.50,
    min_weeks_per_price: float = 1.0
) -> pd.DataFrame:
    """
    Filters products passing adjustable stability thresholds.
    """
    mask = (
        (price_stats_df['Weeks_With_Sales'] >= min_weeks_with_sales) &
        (price_stats_df['Total_Units_Train'] >= min_total_units) &
        (price_stats_df['Min_Max_Price_Ratio'] >= min_price_ratio) &
        (price_stats_df['Weeks_Per_Price'] >= min_weeks_per_price)
    )
    return price_stats_df[mask].sort_values(by='Total_Units_Train', ascending=False).reset_index(drop=True)


WHAT_IF_DISCLAIMER: str = (
    "NOTE: This tool is strictly a what-if revenue scenario calculator (Price × Predicted Quantity), "
    "NOT a causal price elasticity model (dQ/dP). Non-price features are held fixed at their reference values."
)


def calculate_what_if_revenue(
    price: float,
    predicted_quantity: float
) -> float:
    """
    Computes hypothetical revenue for a given price and predicted demand quantity.
    
    Formula:
        Revenue = Price * Predicted_Quantity
        
    Parameters
    ----------
    price : float
        Selling price per unit.
    predicted_quantity : float
        Forecasted weekly demand from the model.
        
    Returns
    -------
    float
        Simulated weekly revenue (clipped at 0 if negative inputs).
    """
    if price < 0 or predicted_quantity < 0:
        return 0.0
    return float(price * predicted_quantity)


def generate_price_scenario_table(
    base_price: float,
    predicted_quantity: float,
    percent_changes: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Generates a comparative sensitivity table across a grid of percentage adjustments.
    Maintained for backward compatibility with existing Streamlit interactive tools.
    """
    if percent_changes is None:
        percent_changes = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]

    base_revenue = calculate_what_if_revenue(base_price, predicted_quantity)
    scenarios: List[Dict[str, Any]] = []

    for pct in percent_changes:
        scenario_price = max(0.01, base_price * (1.0 + pct))
        simulated_revenue = calculate_what_if_revenue(scenario_price, predicted_quantity)
        rev_delta = simulated_revenue - base_revenue

        scenarios.append({
            "Scenario": f"{'+' if pct > 0 else ''}{int(pct * 100)}%",
            "Unit Price (£)": round(scenario_price, 2),
            "Forecasted Demand": round(predicted_quantity, 1),
            "Simulated Revenue (£)": round(simulated_revenue, 2),
            "Revenue Difference (£)": round(rev_delta, 2)
        })

    return pd.DataFrame(scenarios)


def calculate_candidate_price_bounds(
    observed_prices: Union[pd.Series, np.ndarray, List[float]],
    lower_pct: float = 5.0,
    upper_pct: float = 95.0
) -> Tuple[float, float]:
    """
    Computes the lower and upper percentile bounds of observed weekly prices.
    
    Parameters
    ----------
    observed_prices : Union[pd.Series, np.ndarray, List[float]]
        Collection of observed historical weekly prices.
    lower_pct : float
        Lower percentile threshold (default: 5.0).
    upper_pct : float
        Upper percentile threshold (default: 95.0).
        
    Returns
    -------
    Tuple[float, float]
        (p_low, p_high) rounded to 2 decimal places.
    """
    clean_prices = np.asarray(observed_prices, dtype=float)
    clean_prices = clean_prices[~np.isnan(clean_prices) & (clean_prices > 0)]

    if len(clean_prices) == 0:
        raise ValueError("Cannot compute price bounds: no positive observed prices provided.")

    p_low = float(np.percentile(clean_prices, lower_pct))
    p_high = float(np.percentile(clean_prices, upper_pct))
    return round(p_low, 2), round(p_high, 2)


def build_candidate_prices(
    observed_prices: Union[pd.Series, np.ndarray, List[float]],
    n_candidates: int = 10,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0
) -> np.ndarray:
    """
    Builds candidate selling prices strictly inside the 5th to 95th percentile
    of observed weekly prices for a product.
    
    Parameters
    ----------
    observed_prices : Union[pd.Series, np.ndarray, List[float]]
        Historical observed weekly prices for the item.
    n_candidates : int
        Number of candidate prices to generate inside the interval.
    lower_pct : float
        Lower percentile (default 5.0).
    upper_pct : float
        Upper percentile (default 95.0).
        
    Returns
    -------
    np.ndarray
        Array of candidate price points sorted ascending.
    """
    p_low, p_high = calculate_candidate_price_bounds(observed_prices, lower_pct, upper_pct)

    if p_low >= p_high:
        return np.array([p_low])

    prices = np.linspace(p_low, p_high, n_candidates)
    return np.round(prices, 2)


def check_demand_monotonicity(
    candidate_prices: np.ndarray,
    predicted_quantities: np.ndarray,
    tolerance: float = 1e-3,
    stock_code: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Checks whether predicted demand rises as price increases.
    
    Law of Demand expectation:
        dQ/dP <= 0 (Demand non-increasing with respect to price).
        
    If predicted demand increases when price rises (Delta Q > tolerance),
    it flags the anomaly and constructs an informative warning.
    
    Parameters
    ----------
    candidate_prices : np.ndarray
        Array of candidate prices (sorted ascending).
    predicted_quantities : np.ndarray
        Predicted demand quantities corresponding to candidate prices.
    tolerance : float
        Threshold above which an increase is flagged (default 1e-3).
    stock_code : Optional[str]
        Product identifier for warning context.
        
    Returns
    -------
    Tuple[bool, List[str]]
        - demand_rises: True if demand increased on any price increment.
        - warnings_list: List of detailed warning strings.
    """
    demand_rises = False
    warning_messages: List[str] = []
    prod_label = f"Product {stock_code}" if stock_code else "Product"

    for i in range(len(candidate_prices) - 1):
        p_curr = candidate_prices[i]
        p_next = candidate_prices[i + 1]
        q_curr = predicted_quantities[i]
        q_next = predicted_quantities[i + 1]

        price_delta = p_next - p_curr
        qty_delta = q_next - q_curr

        if price_delta > 0 and qty_delta > tolerance:
            demand_rises = True
            msg = (
                f"[{prod_label}] Non-causal demand response detected: "
                f"Predicted demand rises from {q_curr:.1f} to {q_next:.1f} (+{qty_delta:.1f} units) "
                f"when price increases from £{p_curr:.2f} to £{p_next:.2f}."
            )
            warning_messages.append(msg)

    return demand_rises, warning_messages


def predict_demand_with_candidate_prices(
    model: Any,
    feature_row: Union[pd.Series, Dict[str, Any]],
    candidate_prices: np.ndarray,
    feature_cols: List[str],
    price_col: str = 'mean_unit_price'
) -> np.ndarray:
    """
    Predicts demand across candidate prices while holding all other features fixed.
    
    Parameters
    ----------
    model : Any
        Trained scikit-learn regressor (e.g. LinearRegression or RandomForestRegressor).
    feature_row : Union[pd.Series, Dict[str, Any]]
        Reference feature row specifying values for all non-price features:
        ['lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4', 'week_of_year'].
    candidate_prices : np.ndarray
        Array of candidate price points.
    feature_cols : List[str]
        Exact feature column list expected by the model.
    price_col : str
        Name of the price feature column ('mean_unit_price').
        
    Returns
    -------
    np.ndarray
        Array of predicted demand quantities clipped at zero (no negative demand).
    """
    # Replicate reference row across all candidate price points
    candidate_df = pd.DataFrame([feature_row] * len(candidate_prices), columns=feature_cols)
    candidate_df[price_col] = candidate_prices

    # Run model prediction
    raw_preds = model.predict(candidate_df[feature_cols])
    return np.clip(raw_preds, 0.0, None)


def simulate_product_pricing(
    model: Any,
    feature_row: Union[pd.Series, Dict[str, Any]],
    observed_prices: Union[pd.Series, np.ndarray, List[float]],
    feature_cols: List[str],
    price_col: str = 'mean_unit_price',
    n_candidates: int = 10,
    lower_pct: float = 5.0,
    upper_pct: float = 95.0,
    stock_code: Optional[str] = None,
    description: Optional[str] = None,
    emit_warnings: bool = True
) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Performs full pricing simulation for an individual product:
    1. Determines candidate prices inside the 5th to 95th percentile of observed weekly price.
    2. Holds all other features fixed and predicts demand with the model.
    3. Computes revenue = price * predicted quantity.
    4. Evaluates demand monotonicity and warns if demand rises with price.
    
    Parameters
    ----------
    model : Any
        Trained model.
    feature_row : Union[pd.Series, Dict[str, Any]]
        Reference non-price feature values.
    observed_prices : Union[pd.Series, np.ndarray, List[float]]
        Observed historical prices for this product.
    feature_cols : List[str]
        Feature columns expected by model.
    price_col : str
        Price column name ('mean_unit_price').
    n_candidates : int
        Number of candidate prices.
    lower_pct : float
        Lower percentile bound (default 5.0).
    upper_pct : float
        Upper percentile bound (default 95.0).
    stock_code : Optional[str]
        StockCode identifier.
    description : Optional[str]
        Product description.
    emit_warnings : bool
        Whether to emit Python UserWarning when demand rises with price.
        
    Returns
    -------
    Tuple[pd.DataFrame, bool, List[str]]
        - simulation_table: Table with Candidate_Price, Predicted_Demand, Simulated_Revenue, etc.
        - demand_rises: True if predicted demand rises with price anywhere in candidate range.
        - warning_messages: List of formatted warning strings.
    """
    candidate_prices = build_candidate_prices(
        observed_prices, n_candidates=n_candidates, lower_pct=lower_pct, upper_pct=upper_pct
    )

    pred_quantities = predict_demand_with_candidate_prices(
        model, feature_row, candidate_prices, feature_cols, price_col=price_col
    )

    # Check for non-causal demand increases
    demand_rises, warning_msgs = check_demand_monotonicity(
        candidate_prices, pred_quantities, stock_code=stock_code
    )

    if demand_rises and emit_warnings:
        for w in warning_msgs:
            warnings.warn(w, UserWarning)

    # Build scenario table
    records = []
    base_price = float(feature_row[price_col]) if price_col in feature_row else candidate_prices[0]
    base_revenue = None

    for i, (price, qty) in enumerate(zip(candidate_prices, pred_quantities)):
        rev = calculate_what_if_revenue(price, qty)
        prev_qty = pred_quantities[i - 1] if i > 0 else qty
        delta_q = round(qty - prev_qty, 2)
        step_rises = bool(i > 0 and delta_q > 1e-3)

        records.append({
            'StockCode': str(stock_code) if stock_code else '',
            'Description': str(description) if description else '',
            'Candidate_Price': round(price, 2),
            'Predicted_Demand': round(qty, 2),
            'Simulated_Revenue': round(rev, 2),
            'Price_Step_Delta_Q': delta_q if i > 0 else 0.0,
            'Demand_Rises_With_Price': step_rises
        })

    sim_df = pd.DataFrame(records)
    return sim_df, demand_rises, warning_msgs


def run_batch_pricing_simulation_on_passed_products(
    model: Any,
    feature_df: pd.DataFrame,
    test_df: pd.DataFrame,
    passed_products_df: pd.DataFrame,
    feature_cols: List[str],
    price_col: str = 'mean_unit_price',
    product_col: str = 'StockCode',
    desc_col: str = 'description',
    n_candidates: int = 10,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs the what-if pricing simulation across all products that passed the price check
    and are present in the modeled dataset.
    
    Parameters
    ----------
    model : Any
        Trained forecasting model (e.g. LinearRegression or tuned RandomForestRegressor).
    feature_df : pd.DataFrame
        Complete feature panel dataset containing historical observed weekly prices.
    test_df : pd.DataFrame
        Test dataset containing the latest chronological feature states.
    passed_products_df : pd.DataFrame
        Table of products passing the price stability audit.
    feature_cols : List[str]
        Model feature list.
    price_col : str
        Price column name.
    product_col : str
        StockCode column name.
    desc_col : str
        Description column name.
    n_candidates : int
        Number of candidate prices per product.
    verbose : bool
        Whether to print summary progress and warnings.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - detailed_scenarios_df: Full candidate price scenarios across all passed products.
        - summary_df: Per-product summary with optimal revenue, price range, and warning flags.
    """
    passed_codes = passed_products_df[product_col].astype(str).tolist()
    available_codes = feature_df[product_col].astype(str).unique()
    common_codes = [c for c in passed_codes if c in available_codes]

    if verbose:
        print(f"\n[pricing] Running pricing simulation on {len(common_codes)} products passing price check...")

    all_scenarios = []
    summary_records = []
    total_warnings = 0
    products_with_warnings = []

    for sc in common_codes:
        # 1. Historical observed prices for this product
        p_hist = feature_df[feature_df[product_col] == sc]
        observed_prices = p_hist[price_col].dropna().values
        if len(observed_prices) == 0:
            continue

        p_low, p_high = calculate_candidate_price_bounds(observed_prices, 5.0, 95.0)

        # 2. Extract reference features from latest available test observation
        p_test = test_df[test_df[product_col] == sc]
        if p_test.empty:
            p_test = p_hist
        ref_row = p_test.iloc[-1][feature_cols].copy()
        desc = p_test[desc_col].iloc[0] if desc_col in p_test.columns else sc

        # 3. Simulate pricing
        sim_df, demand_rises, warning_msgs = simulate_product_pricing(
            model=model,
            feature_row=ref_row,
            observed_prices=observed_prices,
            feature_cols=feature_cols,
            price_col=price_col,
            n_candidates=n_candidates,
            stock_code=sc,
            description=desc,
            emit_warnings=False
        )

        all_scenarios.append(sim_df)

        if demand_rises:
            total_warnings += len(warning_msgs)
            products_with_warnings.append(sc)
            if verbose:
                print(f"  * WARNING: StockCode {sc} ({desc[:25]}): {warning_msgs[0]}")

        # Find optimal simulated revenue candidate
        best_scenario = sim_df.loc[sim_df['Simulated_Revenue'].idxmax()]
        base_price = float(ref_row[price_col])
        base_demand = float(model.predict(pd.DataFrame([ref_row], columns=feature_cols))[0])
        base_demand = max(0.0, base_demand)
        base_revenue = calculate_what_if_revenue(base_price, base_demand)

        summary_records.append({
            'StockCode': sc,
            'Description': desc,
            'Observed_P5_Price': p_low,
            'Observed_P95_Price': p_high,
            'Reference_Price': round(base_price, 2),
            'Reference_Demand': round(base_demand, 1),
            'Reference_Revenue': round(base_revenue, 2),
            'Optimal_Candidate_Price': best_scenario['Candidate_Price'],
            'Optimal_Predicted_Demand': best_scenario['Predicted_Demand'],
            'Max_Simulated_Revenue': best_scenario['Simulated_Revenue'],
            'Revenue_Uplift': round(best_scenario['Simulated_Revenue'] - base_revenue, 2),
            'Demand_Rises_With_Price': demand_rises,
            'Warning_Count': len(warning_msgs)
        })

    detailed_scenarios_df = pd.concat(all_scenarios, ignore_index=True)
    summary_df = pd.DataFrame(summary_records).sort_values(by='Reference_Revenue', ascending=False).reset_index(drop=True)

    if verbose:
        print(f"[pricing] Completed simulation. {len(products_with_warnings)} / {len(common_codes)} products flagged with rising demand warnings.")

    return detailed_scenarios_df, summary_df


def main():
    """
    Standalone verification runner for pricing module.
    Loads models, simulates candidate pricing on passed products, and exports artifacts.
    """
    print("=" * 78)
    print("ES 335 ML PROJECT: PRICING WHAT-IF REVENUE SIMULATION (5th-95th PERCENTILE)")
    print("=" * 78)
    print(WHAT_IF_DISCLAIMER)
    print("=" * 78)

    # 1. Load data & models
    from src.cleaning import load_raw_data
    from src.features import build_product_week_table, engineer_features, split_chronological_70_15_15
    from src.models import train_linear_regression, tune_random_forest_on_validation

    df_raw = load_raw_data()
    panel_df = build_product_week_table(df_raw, top_n_products=100)
    feat_df = engineer_features(panel_df)
    train_df, val_df, test_df, _, feature_cols, target_col = split_chronological_70_15_15(feat_df)

    lr_model = train_linear_regression(train_df[feature_cols], train_df[target_col])
    rf_model, best_rf_params, _ = tune_random_forest_on_validation(
        train_df[feature_cols], train_df[target_col],
        val_df[feature_cols], val_df[target_col],
        verbose=False
    )

    passed_products_path = "artifacts/price_check_passed_products.csv"
    if not os.path.exists(passed_products_path):
        train_raw, _, _ = get_training_split(df_raw)
        stats = compute_training_price_metrics(train_raw)
        passed_df = filter_products_by_thresholds(stats)
        os.makedirs("artifacts", exist_ok=True)
        passed_df.to_csv(passed_products_path, index=False)
    else:
        passed_df = pd.read_csv(passed_products_path)

    print(f"\n[Setup] Loaded {len(passed_df)} products passing price check.")

    # 2. Simulate with Linear Regression
    print("\n" + "-" * 65)
    print("SIMULATION A: LINEAR REGRESSION (CHOSEN BEST TEST MODEL)")
    print("-" * 65)
    lr_scenarios, lr_summary = run_batch_pricing_simulation_on_passed_products(
        model=lr_model,
        feature_df=feat_df,
        test_df=test_df,
        passed_products_df=passed_df,
        feature_cols=feature_cols,
        verbose=True
    )

    # 3. Simulate with Random Forest
    print("\n" + "-" * 65)
    print("SIMULATION B: RANDOM FOREST (DEMONSTRATING OBSERVATIONAL STEP NON-MONOTONICITIES)")
    print("-" * 65)
    rf_scenarios, rf_summary = run_batch_pricing_simulation_on_passed_products(
        model=rf_model,
        feature_df=feat_df,
        test_df=test_df,
        passed_products_df=passed_df,
        feature_cols=feature_cols,
        verbose=True
    )

    # 4. Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    lr_scenarios.to_csv("artifacts/pricing_scenarios_linear_regression.csv", index=False)
    lr_summary.to_csv("artifacts/pricing_summary_linear_regression.csv", index=False)
    rf_scenarios.to_csv("artifacts/pricing_scenarios_random_forest.csv", index=False)
    rf_summary.to_csv("artifacts/pricing_summary_random_forest.csv", index=False)

    print("\n" + "=" * 78)
    print("COMPARATIVE MONOTONICITY SUMMARY:")
    print("=" * 78)
    lr_warn_count = lr_summary['Demand_Rises_With_Price'].sum()
    rf_warn_count = rf_summary['Demand_Rises_With_Price'].sum()
    print(f"  * Linear Regression Warnings (Demand rises with price): {lr_warn_count} / {len(lr_summary)} products")
    print(f"  * Random Forest Warnings (Demand rises with price):     {rf_warn_count} / {len(rf_summary)} products")
    print("=" * 78)

    print("\nSample Output from Linear Regression Pricing Simulation (Top Revenue Products):")
    disp_cols = ['StockCode', 'Description', 'Observed_P5_Price', 'Observed_P95_Price', 'Optimal_Candidate_Price', 'Optimal_Predicted_Demand', 'Max_Simulated_Revenue', 'Demand_Rises_With_Price']
    print(lr_summary[disp_cols].head(8).to_string(index=False))

    print("\nArtifacts successfully exported:")
    print("  * artifacts/pricing_scenarios_linear_regression.csv")
    print("  * artifacts/pricing_summary_linear_regression.csv")
    print("  * artifacts/pricing_scenarios_random_forest.csv")
    print("  * artifacts/pricing_summary_random_forest.csv")


if __name__ == "__main__":
    main()
