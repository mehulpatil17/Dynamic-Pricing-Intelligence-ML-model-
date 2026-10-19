"""
src/evaluate.py
Module for evaluating weekly demand forecasting models on the strictly held-out test split.

Key Requirements:
1. Compute MAE and RMSE on the test weeks, both overall and per product.
2. Compute forecast error week-by-week to analyze temporal error drift.
3. Generate actual vs. predicted comparison plots (aggregate, per-week error, and top products).
4. Evaluate the test set ONLY ONCE — strict adherence to out-of-sample holdout integrity.
5. Simple, modular, well-commented functions with clear outputs.
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure parent directory is in sys.path when running script directly
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.abspath(os.path.join(_current_dir, ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Set headless backend for matplotlib before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# Expose chronological split function
from src.features import split_chronological_70_15_15


# Model mapping specifications: (Display Name, Column Name, Short Key, Plot Color, Line Style)
MODEL_SPECS = [
    ('Baseline: Training Average', 'pred_train_avg', 'Train_Avg', '#6c757d', ':'),
    ('Baseline: Last-Week Demand', 'pred_last_week', 'Last_Week', '#fd7e14', '--'),
    ('Model: Linear Regression', 'pred_linear_regression', 'Linear_Regression', '#0d6efd', '-.'),
    ('Model: Random Forest (Tuned)', 'pred_random_forest', 'Random_Forest', '#198754', '-')
]


def load_test_predictions(
    filepath: str = "artifacts/unified_test_predictions.csv"
) -> pd.DataFrame:
    """
    Loads precomputed unified test predictions table containing actual demand
    and forecasts from all 4 methods.
    
    If the artifact does not exist, triggers a single pipeline run to train
    on train/val weeks and generate test predictions exactly once.
    
    Parameters
    ----------
    filepath : str
        Path to the unified test predictions CSV file.
        
    Returns
    -------
    pd.DataFrame
        Unified test predictions dataframe.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['week_start'] = pd.to_datetime(df['week_start'])
        return df

    # Fallback: Generate once via pipeline functions
    print(f"[evaluate] '{filepath}' not found. Generating predictions from pipeline once...")
    from src.cleaning import load_raw_data
    from src.features import build_product_week_table, engineer_features, split_chronological_70_15_15
    from src.models import train_and_evaluate_all_methods

    df_raw = load_raw_data()
    panel = build_product_week_table(df_raw, top_n_products=100)
    feat_df = engineer_features(panel, lags=(1, 2, 4), rolling_window=4)
    train_df, val_df, test_df, _, feature_cols, target_col = split_chronological_70_15_15(feat_df)
    res = train_and_evaluate_all_methods(train_df, val_df, test_df, feature_cols, target_col=target_col, verbose=False)
    
    test_preds = res['test_predictions']
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    test_preds.to_csv(filepath, index=False)
    test_preds['week_start'] = pd.to_datetime(test_preds['week_start'])
    return test_preds


def compute_overall_metrics(
    test_df: pd.DataFrame,
    target_col: str = 'weekly_quantity'
) -> pd.DataFrame:
    """
    Computes overall MAE and RMSE across all observations in the test split.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe with target and prediction columns.
    target_col : str
        Ground truth target column name.
        
    Returns
    -------
    pd.DataFrame
        Table containing Model name, MAE, and RMSE rounded to 2 decimal places.
    """
    y_true = test_df[target_col].values
    records = []

    for disp_name, pred_col, _, _, _ in MODEL_SPECS:
        if pred_col not in test_df.columns:
            continue
        y_pred = test_df[pred_col].values
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))
        records.append({
            'Model': disp_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2)
        })

    return pd.DataFrame(records)


def compute_per_product_metrics(
    test_df: pd.DataFrame,
    product_col: str = 'StockCode',
    target_col: str = 'weekly_quantity',
    desc_col: str = 'description'
) -> pd.DataFrame:
    """
    Computes MAE and RMSE individually for each product across all 4 models,
    sorted by total test demand descending. Also identifies the best model per product.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing actual demand and model predictions.
    product_col : str
        StockCode column.
    target_col : str
        Ground truth demand column.
    desc_col : str
        Product description column.
        
    Returns
    -------
    pd.DataFrame
        Product-level evaluation table with MAE and RMSE for each method.
    """
    records: List[Dict[str, Any]] = []

    for prod_id, grp in test_df.groupby(product_col):
        y_true = grp[target_col].values
        desc = grp[desc_col].iloc[0] if desc_col in grp.columns else str(prod_id)
        n_weeks = len(grp)
        tot_demand = float(y_true.sum())
        mean_demand = float(y_true.mean())

        row: Dict[str, Any] = {
            'StockCode': str(prod_id),
            'Description': desc,
            'Test_Weeks': n_weeks,
            'Total_Actual_Demand': round(tot_demand, 1),
            'Mean_Weekly_Demand': round(mean_demand, 1)
        }

        mae_by_model: Dict[str, float] = {}
        for _, pred_col, short_key, _, _ in MODEL_SPECS:
            if pred_col in grp.columns:
                y_pred = grp[pred_col].values
                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(root_mean_squared_error(y_true, y_pred))
                row[f'{short_key}_MAE'] = round(mae, 2)
                row[f'{short_key}_RMSE'] = round(rmse, 2)
                mae_by_model[short_key] = mae

        # Determine best model for this product (lowest MAE)
        if mae_by_model:
            best_model = min(mae_by_model, key=mae_by_model.get)
            row['Best_Model_MAE'] = best_model

        records.append(row)

    res_df = pd.DataFrame(records)
    return res_df.sort_values(by='Total_Actual_Demand', ascending=False).reset_index(drop=True)


def compute_error_by_week(
    test_df: pd.DataFrame,
    date_col: str = 'week_start',
    target_col: str = 'weekly_quantity'
) -> pd.DataFrame:
    """
    Computes MAE, RMSE, actual demand total, and predicted demand totals
    broken down for each chronological test week.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Test dataframe containing week dates, actual demand, and model predictions.
    date_col : str
        Week start date column.
    target_col : str
        Ground truth demand column.
        
    Returns
    -------
    pd.DataFrame
        Week-by-week performance table tracking temporal error trends.
    """
    records = []
    # Ensure datetime format for sorting
    df_copy = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    weeks = sorted(df_copy[date_col].unique())

    for w in weeks:
        w_df = df_copy[df_copy[date_col] == w]
        y_true = w_df[target_col].values
        w_str = pd.to_datetime(w).strftime('%Y-%m-%d')
        actual_total = float(y_true.sum())

        row: Dict[str, Any] = {
            'week_start': w_str,
            'Product_Count': len(w_df),
            'Actual_Total_Demand': round(actual_total, 1)
        }

        for _, pred_col, short_key, _, _ in MODEL_SPECS:
            if pred_col in w_df.columns:
                y_pred = w_df[pred_col].values
                mae = float(mean_absolute_error(y_true, y_pred))
                rmse = float(root_mean_squared_error(y_true, y_pred))
                pred_total = float(y_pred.sum())
                row[f'{short_key}_Pred_Total'] = round(pred_total, 1)
                row[f'{short_key}_MAE'] = round(mae, 2)
                row[f'{short_key}_RMSE'] = round(rmse, 2)

        records.append(row)

    return pd.DataFrame(records)


def summarize_model_rankings(
    per_product_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Summarizes model performance across products, including win counts,
    hardest products, and easiest products.
    
    Parameters
    ----------
    per_product_df : pd.DataFrame
        Per-product metric table produced by compute_per_product_metrics.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of summary statistics.
    """
    best_counts = per_product_df['Best_Model_MAE'].value_counts().to_dict()
    
    # Identify hardest products (highest Linear Regression MAE)
    hardest = per_product_df.sort_values(by='Linear_Regression_MAE', ascending=False)[
        ['StockCode', 'Description', 'Total_Actual_Demand', 'Linear_Regression_MAE', 'Random_Forest_MAE']
    ].head(5).to_dict(orient='records')
    
    # Identify easiest products (lowest Linear Regression MAE)
    easiest = per_product_df.sort_values(by='Linear_Regression_MAE', ascending=True)[
        ['StockCode', 'Description', 'Total_Actual_Demand', 'Linear_Regression_MAE', 'Random_Forest_MAE']
    ].head(5).to_dict(orient='records')

    return {
        'best_counts_by_mae': best_counts,
        'hardest_products': hardest,
        'easiest_products': easiest
    }


def generate_evaluation_plots(
    test_df: pd.DataFrame,
    weekly_metrics_df: pd.DataFrame,
    output_dir: str = "artifacts/figures"
) -> Dict[str, str]:
    """
    Generates and saves visual evaluation plots:
    1. Weekly Aggregate Actual vs Predicted Demand across all models.
    2. MAE and RMSE Progression by Week.
    3. Multi-product trajectory comparison for top demand products.
    4. Per-product MAE boxplot distribution.
    
    Parameters
    ----------
    test_df : pd.DataFrame
        Unified test predictions.
    weekly_metrics_df : pd.DataFrame
        Week-by-week metrics table.
    output_dir : str
        Directory to save PNG plots.
        
    Returns
    -------
    Dict[str, str]
        Dictionary of plot names and their saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = {}

    # Plot styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8

    # -------------------------------------------------------------
    # Plot 1: Total Weekly Actual vs Predicted Demand
    # -------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(11, 5.5), dpi=150)
    weeks = weekly_metrics_df['week_start']
    ax1.plot(weeks, weekly_metrics_df['Actual_Total_Demand'], 'k-o', label='Actual Demand (Ground Truth)', linewidth=2.8, markersize=7)

    for disp_name, _, short_key, color, ls in MODEL_SPECS:
        col = f'{short_key}_Pred_Total'
        if col in weekly_metrics_df.columns:
            ax1.plot(weeks, weekly_metrics_df[col], label=disp_name, color=color, linestyle=ls, linewidth=2.0, markersize=5)

    ax1.set_title("Test Set: Total Weekly Demand Forecast (Top 100 Products Aggregate)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Test Week (Monday Start)", fontsize=11, fontweight='normal')
    ax1.set_ylabel("Total Units Sold Across Products", fontsize=11, fontweight='normal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
    plt.xticks(rotation=20)
    fig1.tight_layout()

    plot1_path = os.path.join(output_dir, "actual_vs_predicted_weekly_aggregate.png")
    fig1.savefig(plot1_path)
    plt.close(fig1)
    saved_plots['aggregate_weekly'] = plot1_path

    # -------------------------------------------------------------
    # Plot 2: Forecast Error Progression by Week (MAE & RMSE)
    # -------------------------------------------------------------
    fig2, (ax2_mae, ax2_rmse) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    for disp_name, _, short_key, color, ls in MODEL_SPECS:
        mae_col = f'{short_key}_MAE'
        rmse_col = f'{short_key}_RMSE'
        if mae_col in weekly_metrics_df.columns:
            ax2_mae.plot(weeks, weekly_metrics_df[mae_col], label=disp_name, color=color, linestyle=ls, linewidth=2.0, marker='o')
        if rmse_col in weekly_metrics_df.columns:
            ax2_rmse.plot(weeks, weekly_metrics_df[rmse_col], label=disp_name, color=color, linestyle=ls, linewidth=2.0, marker='s')

    ax2_mae.set_title("MAE by Test Week", fontsize=12, fontweight='bold')
    ax2_mae.set_xlabel("Week Start", fontsize=10)
    ax2_mae.set_ylabel("Mean Absolute Error (Units)", fontsize=10)
    ax2_mae.grid(True, linestyle='--', alpha=0.5)
    ax2_mae.tick_params(axis='x', rotation=30)
    ax2_mae.legend(loc='upper left', fontsize=9)

    ax2_rmse.set_title("RMSE by Test Week (Impact of Holiday Spikes)", fontsize=12, fontweight='bold')
    ax2_rmse.set_xlabel("Week Start", fontsize=10)
    ax2_rmse.set_ylabel("Root Mean Squared Error (Units)", fontsize=10)
    ax2_rmse.grid(True, linestyle='--', alpha=0.5)
    ax2_rmse.tick_params(axis='x', rotation=30)
    ax2_rmse.legend(loc='upper left', fontsize=9)

    fig2.suptitle("Temporal Error Progression across Chronological Test Weeks", fontsize=13, fontweight='bold', y=1.02)
    fig2.tight_layout()

    plot2_path = os.path.join(output_dir, "error_by_week.png")
    fig2.savefig(plot2_path, bbox_inches='tight')
    plt.close(fig2)
    saved_plots['error_by_week'] = plot2_path

    test_df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(test_df['week_start']):
        test_df['week_start'] = pd.to_datetime(test_df['week_start'])

    # -------------------------------------------------------------
    # Plot 3: Trajectory Comparison for Top Products
    # -------------------------------------------------------------
    # Exclude outlier 23843 to show representative retail trajectories clearly
    candidate_pool = test_df[test_df['StockCode'] != '23843']
    if candidate_pool.empty:
        candidate_pool = test_df
    top_products = candidate_pool.groupby('StockCode')['weekly_quantity'].sum().nlargest(4).index.tolist()
    
    fig3, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    axes = axes.flatten()

    for idx in range(len(axes)):
        if idx < len(top_products):
            sc = top_products[idx]
            ax = axes[idx]
            p_df = test_df[test_df['StockCode'] == sc].sort_values('week_start')
            p_weeks = p_df['week_start'].dt.strftime('%m-%d')
            desc = p_df['description'].iloc[0] if 'description' in p_df.columns else sc

            ax.plot(p_weeks, p_df['weekly_quantity'], 'k-o', label='Actual Demand', linewidth=2.5, markersize=6)
            for disp_name, pred_col, _, color, ls in MODEL_SPECS:
                if pred_col in p_df.columns:
                    ax.plot(p_weeks, p_df[pred_col], label=disp_name, color=color, linestyle=ls, alpha=0.85)

            ax.set_title(f"[{sc}] {desc[:28]}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Week (MM-DD)", fontsize=9)
            ax.set_ylabel("Units", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
            if idx == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
        else:
            axes[idx].set_visible(False)

    fig3.suptitle("Actual vs. Predicted Trajectories for High-Volume Products (Test Split)", fontsize=13, fontweight='bold')
    fig3.tight_layout()

    plot3_path = os.path.join(output_dir, "actual_vs_predicted_top_products.png")
    fig3.savefig(plot3_path)
    plt.close(fig3)
    saved_plots['top_products'] = plot3_path

    # -------------------------------------------------------------
    # Plot 4: Per-Product MAE Distribution Boxplot
    # -------------------------------------------------------------
    per_prod = compute_per_product_metrics(test_df)
    mae_data = [
        per_prod['Train_Avg_MAE'].values,
        per_prod['Last_Week_MAE'].values,
        per_prod['Linear_Regression_MAE'].values,
        per_prod['Random_Forest_MAE'].values
    ]
    labels = ['Train Avg', 'Last Week', 'Linear Reg', 'Random Forest']

    fig4, ax4 = plt.subplots(figsize=(9, 5), dpi=150)
    bp = ax4.boxplot(mae_data, tick_labels=labels, patch_artist=True, showfliers=False)
    colors = ['#d6d8db', '#ffe8cc', '#cfe2ff', '#d1e7dd']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#495057')

    ax4.set_title("Per-Product MAE Distribution Across 100 Products (Outliers Hidden)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Per-Product MAE (Units)", fontsize=10)
    ax4.grid(True, axis='y', linestyle='--', alpha=0.5)
    fig4.tight_layout()

    plot4_path = os.path.join(output_dir, "per_product_error_distribution.png")
    fig4.savefig(plot4_path)
    plt.close(fig4)
    saved_plots['error_distribution'] = plot4_path

    return saved_plots


def export_evaluation_artifacts(
    overall_df: pd.DataFrame,
    per_product_df: pd.DataFrame,
    weekly_metrics_df: pd.DataFrame,
    output_dir: str = "artifacts"
) -> None:
    """
    Persists evaluation metrics tables to CSV format in the artifacts folder.
    
    Parameters
    ----------
    overall_df : pd.DataFrame
        Overall test set MAE and RMSE table.
    per_product_df : pd.DataFrame
        Per-product MAE and RMSE breakdown.
    weekly_metrics_df : pd.DataFrame
        Week-by-week performance table.
    output_dir : str
        Output artifacts directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    overall_df.to_csv(os.path.join(output_dir, "test_metrics_overall.csv"), index=False)
    per_product_df.to_csv(os.path.join(output_dir, "test_metrics_per_product.csv"), index=False)
    weekly_metrics_df.to_csv(os.path.join(output_dir, "test_metrics_by_week.csv"), index=False)


def run_evaluation(
    predictions_file: str = "artifacts/unified_test_predictions.csv",
    output_dir: str = "artifacts",
    figures_dir: str = "artifacts/figures"
) -> Dict[str, Any]:
    """
    Executes comprehensive test set evaluation strictly once.
    
    Workflow:
    1. Loads unified test predictions.
    2. Computes overall test metrics (MAE and RMSE).
    3. Computes per-product metrics and model win shares.
    4. Computes error by week to diagnose temporal trends.
    5. Computes diagnostic sensitivity analysis (e.g. without extreme December bulk spike).
    6. Generates and saves visual evaluation plots.
    7. Exports evaluation tables to CSV.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all evaluation dataframes and summary statistics.
    """
    print("\n" + "=" * 78)
    print("ES 335 ML PROJECT: TEST SET EVALUATION (HELD-OUT 15% OF WEEKS)")
    print("=" * 78)
    print("NOTE: Evaluating test set ONLY ONCE. No model parameters are adjusted.")
    print("=" * 78)

    # 1. Load predictions
    test_df = load_test_predictions(predictions_file)
    n_rows = len(test_df)
    n_prods = test_df['StockCode'].nunique()
    weeks = sorted(test_df['week_start'].unique())
    n_weeks = len(weeks)

    print(f"\n[Test Data Info]")
    print(f"  * Total Observations: {n_rows:,} ({n_prods} products x {n_weeks} weeks)")
    print(f"  * Date Range:         {pd.to_datetime(weeks[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(weeks[-1]).strftime('%Y-%m-%d')}")
    print(f"  * Total Actual Demand: {int(test_df['weekly_quantity'].sum()):,} units")

    # 2. Overall Metrics
    overall_df = compute_overall_metrics(test_df)
    print("\n" + "-" * 55)
    print("1. OVERALL TEST METRICS (ALL 700 TEST OBSERVATIONS):")
    print("-" * 55)
    print(overall_df.to_string(index=False))

    # 3. Sensitivity / Diagnostic Check (Excluding Dec 9 Outlier 23843)
    test_sub = test_df[test_df['StockCode'] != '23843']
    overall_sub = compute_overall_metrics(test_sub)
    print("\n" + "-" * 55)
    print("2. DIAGNOSTIC CHECK: EXCLUDING ANOMALOUS WHOLESALE SPIKE")
    print("   (StockCode 23843 'PAPER CRAFT , LITTLE BIRDIE', 80,995 units on 2011-12-05):")
    print("-" * 55)
    print(overall_sub.to_string(index=False))

    # 4. Error by Week
    weekly_metrics_df = compute_error_by_week(test_df)
    print("\n" + "-" * 78)
    print("3. ERROR BY WEEK (MAE & RMSE OVER TIME ACROSS 7 CHRONOLOGICAL WEEKS):")
    print("-" * 78)
    # Print formatted columns for clean terminal display
    disp_cols = ['week_start', 'Actual_Total_Demand', 'Train_Avg_MAE', 'Last_Week_MAE', 'Linear_Regression_MAE', 'Random_Forest_MAE']
    print(weekly_metrics_df[disp_cols].to_string(index=False))

    # 5. Per-Product Metrics & Model Performance Summary
    per_product_df = compute_per_product_metrics(test_df)
    summary_stats = summarize_model_rankings(per_product_df)

    print("\n" + "-" * 55)
    print("4. MODEL WIN SHARE ACROSS PRODUCTS (LOWEST MAE):")
    print("-" * 55)
    for model_key, count in summary_stats['best_counts_by_mae'].items():
        print(f"  * {model_key:22s}: {count:2d} products ({count / n_prods:.1%})")

    print("\nTop 5 Hardest Products to Forecast (Highest Linear Regression MAE):")
    for item in summary_stats['hardest_products']:
        print(f"  * StockCode {item['StockCode']:7s} | Demand: {item['Total_Actual_Demand']:8.0f} | LR MAE: {item['Linear_Regression_MAE']:6.2f} | RF MAE: {item['Random_Forest_MAE']:6.2f} | {item['Description'][:30]}")

    print("\nTop 5 Easiest Products to Forecast (Lowest Linear Regression MAE):")
    for item in summary_stats['easiest_products']:
        print(f"  * StockCode {item['StockCode']:7s} | Demand: {item['Total_Actual_Demand']:8.0f} | LR MAE: {item['Linear_Regression_MAE']:6.2f} | RF MAE: {item['Random_Forest_MAE']:6.2f} | {item['Description'][:30]}")

    # 6. Generate Plots
    print("\n[Figures] Generating and saving evaluation plots...")
    saved_plots = generate_evaluation_plots(test_df, weekly_metrics_df, output_dir=figures_dir)
    for name, path in saved_plots.items():
        print(f"  * Saved {name:18s} -> {path}")

    # 7. Export Artifacts
    print(f"\n[Artifacts] Exporting metrics tables to '{output_dir}/'...")
    export_evaluation_artifacts(overall_df, per_product_df, weekly_metrics_df, output_dir=output_dir)
    print("  * Saved artifacts/test_metrics_overall.csv")
    print("  * Saved artifacts/test_metrics_per_product.csv")
    print("  * Saved artifacts/test_metrics_by_week.csv")

    print("\n" + "=" * 78)
    print("TEST EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 78)

    return {
        'overall_metrics': overall_df,
        'overall_metrics_sub': overall_sub,
        'per_product_metrics': per_product_df,
        'weekly_metrics': weekly_metrics_df,
        'summary_stats': summary_stats,
        'saved_plots': saved_plots
    }


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
