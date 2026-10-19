"""
src/run_pipeline.py
End-to-end pipeline execution module for ES 335 ML Project.
Loads data, engineers features, splits chronologically 70/15/15, trains models,
tunes Random Forest on validation weeks, evaluates test set, runs pricing simulation,
and writes all predictions and metrics to disk.
"""

import os
import sys
import time
import pandas as pd

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.cleaning import load_raw_data, get_data_summary
from src.features import (
    build_product_week_table,
    engineer_features,
    split_chronological_70_15_15
)
from src.models import train_and_evaluate_all_methods
from src.evaluate import compute_per_product_metrics


def main():
    t_start = time.time()
    print("=" * 75)
    print("ES 335 ML Project: Weekly Demand Forecasting Pipeline")
    print("=" * 75)

    # 1. Load Data
    print("\n[Step 1/6] Loading cleaned dataset...")
    raw_path = os.path.join(_project_root, "data/raw/Online Retail.xlsx")
    cache_path = os.path.join(_project_root, "data/processed/data_clean.parquet")
    df_raw = load_raw_data(filepath=raw_path, cache_path=cache_path)
    summary = get_data_summary(df_raw)
    print(f"  * Total valid records: {summary['total_records']:,}")
    print(f"  * Unique products:     {summary['unique_products']:,}")
    print(f"  * Date range:          {summary['date_min']} to {summary['date_max']}")

    # 2. Build Continuous Product-Week Table
    print("\n[Step 2/6] Building continuous product-week table for top 100 products...")
    panel_df = build_product_week_table(df_raw, top_n_products=100)
    proc_dir = os.path.join(_project_root, "data/processed")
    os.makedirs(proc_dir, exist_ok=True)
    panel_path = os.path.join(proc_dir, "product_week_table.parquet")
    panel_df.to_parquet(panel_path, index=False)
    n_weeks = panel_df['week_start'].nunique()
    n_prods = panel_df['StockCode'].nunique()
    n_zeros = (panel_df['weekly_quantity'] == 0).sum()
    print(f"  * Balanced panel shape:    {panel_df.shape} ({n_prods} products x {n_weeks} weeks)")
    print(f"  * Zero-demand weeks filled: {n_zeros:,} / {len(panel_df):,} ({n_zeros / len(panel_df):.1%})")
    print(f"  * Saved product-week table to '{panel_path}'")

    # 3. Engineer Causal Features
    print("\n[Step 3/6] Engineering lag 1/2/4, shifted rolling mean 4, and calendar features...")
    feature_df = engineer_features(panel_df, lags=(1, 2, 4), rolling_window=4)
    print(f"  * Feature dataset shape: {feature_df.shape} (after dropping initial 4-week warm-up)")

    # 4. Strict Chronological 70/15/15 Split
    print("\n[Step 4/6] Performing strict chronological 70/15/15 split...")
    train_df, val_df, test_df, split_info, feature_cols, target_col = split_chronological_70_15_15(feature_df)
    print(f"  * Train: {split_info['train_weeks_count']} weeks ({split_info['train_date_range'][0].strftime('%Y-%m-%d')} to {split_info['train_date_range'][1].strftime('%Y-%m-%d')}) | {split_info['train_rows']:,} rows")
    print(f"  * Val:   {split_info['val_weeks_count']} weeks ({split_info['val_date_range'][0].strftime('%Y-%m-%d')} to {split_info['val_date_range'][1].strftime('%Y-%m-%d')}) | {split_info['val_rows']:,} rows")
    print(f"  * Test:  {split_info['test_weeks_count']} weeks ({split_info['test_date_range'][0].strftime('%Y-%m-%d')} to {split_info['test_date_range'][1].strftime('%Y-%m-%d')}) | {split_info['test_rows']:,} rows")
    print(f"  * Features ({len(feature_cols)}): {feature_cols}")

    # 5. Train, Tune & Evaluate All Methods
    print("\n[Step 5/6] Training Baselines, Linear Regression, and tuning Random Forest on validation weeks...")
    pipeline_results = train_and_evaluate_all_methods(
        train_df, val_df, test_df, feature_cols, target_col=target_col, verbose=True
    )

    val_metrics = pipeline_results['val_metrics']
    test_metrics = pipeline_results['test_metrics']
    val_table = pipeline_results['val_predictions']
    test_table = pipeline_results['test_predictions']

    print("=" * 75)
    print("VERIFIED EVALUATION METRICS ON VALIDATION SPLIT (15% of weeks):")
    print("=" * 75)
    print(val_metrics.to_string(index=False))

    print("\n" + "=" * 75)
    print("VERIFIED EVALUATION METRICS ON TEST SPLIT (15% of weeks):")
    print("=" * 75)
    print(test_metrics.to_string(index=False))
    print("=" * 75)

    # Save artifacts to artifacts/
    art_dir = os.path.join(_project_root, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    test_table.to_csv(os.path.join(art_dir, "unified_test_predictions.csv"), index=False)
    test_table.to_parquet(os.path.join(art_dir, "test_with_preds.parquet"), index=False)
    val_table.to_csv(os.path.join(art_dir, "unified_val_predictions.csv"), index=False)
    test_metrics.to_csv(os.path.join(art_dir, "model_results.csv"), index=False)
    val_metrics.to_csv(os.path.join(art_dir, "val_model_results.csv"), index=False)
    pipeline_results['rf_tuning_history'].to_csv(os.path.join(art_dir, "rf_tuning_history.csv"), index=False)
    panel_df.to_parquet(os.path.join(art_dir, "panel_full.parquet"), index=False)

    # Feature weights & importances
    lr_model = pipeline_results['models']['linear_regression']
    rf_model = pipeline_results['models']['random_forest']
    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Importance': rf_model.feature_importances_,
        'LR_Coefficient': lr_model.coef_
    }).sort_values(by='RF_Importance', ascending=False)
    feat_df.to_csv(os.path.join(art_dir, "feature_importance.csv"), index=False)

    # Step 6: Comprehensive Test Evaluation & Figures Export
    print("\n" + "=" * 75)
    print("[Step 6/6] Executing test evaluation & generating publication figures...")
    print("=" * 75)
    from src.evaluate import run_evaluation
    eval_res = run_evaluation(
        predictions_file=os.path.join(art_dir, "unified_test_predictions.csv"),
        output_dir=art_dir,
        figures_dir=os.path.join(art_dir, "figures")
    )

    # Also copy figures to reports/figures/
    reports_fig_dir = os.path.join(_project_root, "reports/figures")
    os.makedirs(reports_fig_dir, exist_ok=True)
    for fig_file in os.listdir(os.path.join(art_dir, "figures")):
        src_f = os.path.join(art_dir, "figures", fig_file)
        dst_f = os.path.join(reports_fig_dir, fig_file)
        if os.path.isfile(src_f):
            import shutil
            shutil.copy2(src_f, dst_f)

    # Step 7: Pricing Simulation on Passed Products
    print("\n" + "=" * 75)
    print("[Pricing Simulation] Running candidate pricing inside 5th-95th percentile on passed products...")
    print("=" * 75)
    from src.pricing import run_batch_pricing_simulation_on_passed_products
    passed_products_path = os.path.join(art_dir, "price_check_passed_products.csv")
    if os.path.exists(passed_products_path):
        passed_df = pd.read_csv(passed_products_path)
    else:
        from src.pricing import compute_training_price_metrics, filter_products_by_thresholds, get_training_split
        train_raw, _, _ = get_training_split(df_raw)
        stats = compute_training_price_metrics(train_raw)
        passed_df = filter_products_by_thresholds(stats)
        passed_df.to_csv(passed_products_path, index=False)

    lr_scenarios, lr_summary = run_batch_pricing_simulation_on_passed_products(
        model=lr_model, feature_df=feature_df,
        test_df=test_df, passed_products_df=passed_df, feature_cols=feature_cols, verbose=True
    )
    lr_scenarios.to_csv(os.path.join(art_dir, "pricing_scenarios_linear_regression.csv"), index=False)
    lr_summary.to_csv(os.path.join(art_dir, "pricing_summary_linear_regression.csv"), index=False)

    rf_scenarios, rf_summary = run_batch_pricing_simulation_on_passed_products(
        model=rf_model, feature_df=feature_df,
        test_df=test_df, passed_products_df=passed_df, feature_cols=feature_cols, verbose=True
    )
    rf_scenarios.to_csv(os.path.join(art_dir, "pricing_scenarios_random_forest.csv"), index=False)
    rf_summary.to_csv(os.path.join(art_dir, "pricing_summary_random_forest.csv"), index=False)

    print(f"\n[Artifacts] Successfully wrote all predictions, metrics, and pricing simulations to '{art_dir}/'")
    elapsed = time.time() - t_start
    print(f"Pipeline finished successfully in {elapsed:.2f} seconds!")


if __name__ == "__main__":
    main()
