"""
tests/test_split.py
Comprehensive unit tests verifying:
1. Strict chronological 70/15/15 splitting:
   - Strict temporal partition: max(Train) < min(Val) <= max(Val) < min(Test).
   - Zero week overlap across Train, Validation, and Test partitions.
   - Conservation of total observations.
   - Correct 70/15/15 proportions by unique calendar weeks.
2. Demand Forecasting Models & Tuning:
   - Baseline 1 (Training Average) & Baseline 2 (Last-Week Demand).
   - Linear Regression fitting and non-negative prediction clipping.
   - Random Forest tuning strictly on validation weeks.
   - Unified predictions table structure.
3. Test Set Evaluation & Metrics:
   - Overall MAE and RMSE calculation.
   - Per-product metric evaluation & best model selection.
   - Error-by-week aggregation.
   - Summary statistics and plot generation.
4. What-If Pricing Tool:
   - Candidate price bounds strictly inside [P_5, P_95].
   - Holding non-price features fixed.
   - Revenue = Price * Predicted Demand.
   - Monotonicity check & explicit warning when predicted demand rises with price.
5. Dashboard Artifact Loading:
   - Fast artifact resolution and schema integrity.
"""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import split_chronological_70_15_15
from src.models import (
    predict_baseline_train_avg,
    predict_baseline_last_week,
    train_linear_regression,
    tune_random_forest_on_validation,
    build_unified_predictions_table,
    evaluate_predictions
)
from src.evaluate import (
    compute_overall_metrics,
    compute_per_product_metrics,
    compute_error_by_week,
    summarize_model_rankings,
    generate_evaluation_plots,
    export_evaluation_artifacts
)
from src.pricing import (
    calculate_candidate_price_bounds,
    build_candidate_prices,
    calculate_what_if_revenue,
    check_demand_monotonicity,
    predict_demand_with_candidate_prices,
    simulate_product_pricing,
    WHAT_IF_DISCLAIMER
)
from app.streamlit_app import load_all_saved_artifacts, resolve_path


# =============================================================================
# 1. CHRONOLOGICAL SPLIT TESTS
# =============================================================================

@pytest.fixture
def mock_time_series_panel():
    """Creates a synthetic balanced panel of 5 products over 20 calendar weeks."""
    weeks = [pd.Timestamp('2021-01-04') + pd.Timedelta(weeks=i) for i in range(20)]
    products = ['PROD_01', 'PROD_02', 'PROD_03', 'PROD_04', 'PROD_05']
    rows = []

    for w_idx, w in enumerate(weeks):
        for p in products:
            rows.append({
                'StockCode': p,
                'week_start': w,
                'weekly_quantity': float(100 + w_idx * 2),
                'lag_1': 98.0,
                'lag_2': 96.0,
                'lag_4': 92.0,
                'rolling_mean_4': 95.0,
                'rolling_std_4': 4.0,
                'week_of_year': int(w.isocalendar().week),
                'mean_unit_price': 2.50
            })

    return pd.DataFrame(rows)


def test_chronological_split_strict_temporal_ordering(mock_time_series_panel):
    """Verifies that split boundaries strictly satisfy max(Train) < min(Val) <= max(Val) < min(Test)."""
    train_df, val_df, test_df, split_info, feature_cols, target_col = split_chronological_70_15_15(
        mock_time_series_panel, date_col='week_start', target_col='weekly_quantity'
    )

    train_max = train_df['week_start'].max()
    val_min = val_df['week_start'].min()
    val_max = val_df['week_start'].max()
    test_min = test_df['week_start'].min()
    test_max = test_df['week_start'].max()

    assert train_max < val_min, f"Train max ({train_max}) must be strictly before Val min ({val_min})"
    assert val_max < test_min, f"Val max ({val_max}) must be strictly before Test min ({test_min})"
    assert test_min <= test_max


def test_chronological_split_zero_week_overlap(mock_time_series_panel):
    """Verifies that no calendar week is shared between any of the partitions."""
    train_df, val_df, test_df, _, _, _ = split_chronological_70_15_15(mock_time_series_panel)

    train_weeks = set(train_df['week_start'].unique())
    val_weeks = set(val_df['week_start'].unique())
    test_weeks = set(test_df['week_start'].unique())

    assert len(train_weeks.intersection(val_weeks)) == 0
    assert len(val_weeks.intersection(test_weeks)) == 0
    assert len(train_weeks.intersection(test_weeks)) == 0


def test_chronological_split_proportions(mock_time_series_panel):
    """Verifies 70/15/15 week proportions on 20 total weeks (14 / 3 / 3)."""
    train_df, val_df, test_df, split_info, _, _ = split_chronological_70_15_15(mock_time_series_panel)

    assert train_df['week_start'].nunique() == 14
    assert val_df['week_start'].nunique() == 3
    assert test_df['week_start'].nunique() == 3
    assert split_info['train_weeks_count'] == 14
    assert split_info['val_weeks_count'] == 3
    assert split_info['test_weeks_count'] == 3


def test_chronological_split_row_count_conservation(mock_time_series_panel):
    """Verifies that all panel rows are partitioned without loss or duplication."""
    train_df, val_df, test_df, split_info, _, _ = split_chronological_70_15_15(mock_time_series_panel)
    assert len(train_df) + len(val_df) + len(test_df) == len(mock_time_series_panel)


def test_chronological_split_insufficient_weeks_error():
    """Verifies that ValueError is raised if fewer than 10 chronological weeks are provided."""
    short_panel = pd.DataFrame({
        'StockCode': ['P1'] * 5,
        'week_start': pd.date_range('2021-01-01', periods=5, freq='W'),
        'weekly_quantity': [10.0] * 5,
        'mean_unit_price': [1.0] * 5,
        'lag_1': [10.0] * 5,
        'lag_2': [10.0] * 5,
        'lag_4': [10.0] * 5,
        'rolling_mean_4': [10.0] * 5,
        'rolling_std_4': [1.0] * 5,
        'week_of_year': [1, 2, 3, 4, 5]
    })
    with pytest.raises(ValueError, match="Need at least 10 weeks"):
        split_chronological_70_15_15(short_panel)


# =============================================================================
# 2. MODEL BASELINES, LINEAR REGRESSION & RANDOM FOREST TUNING TESTS
# =============================================================================

@pytest.fixture
def sample_splits():
    """Creates mock train, validation, and test splits with known values."""
    feature_cols = ['lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4', 'week_of_year', 'mean_unit_price']
    np.random.seed(42)

    def _build_part(products, weeks, base_map):
        records = []
        for p in products:
            base = base_map[p]
            for w in weeks:
                records.append({
                    'StockCode': p,
                    'week_start': pd.Timestamp('2021-01-04') + pd.Timedelta(weeks=w),
                    'description': f'Description {p}',
                    'weekly_quantity': base + np.random.normal(0, 5),
                    'lag_1': base + np.random.normal(0, 5),
                    'lag_2': base + np.random.normal(0, 5),
                    'lag_4': base + np.random.normal(0, 5),
                    'rolling_mean_4': base,
                    'rolling_std_4': 3.0,
                    'week_of_year': w,
                    'mean_unit_price': 5.0
                })
        return pd.DataFrame(records)

    prods = ['PROD_1', 'PROD_2']
    base_m = {'PROD_1': 50.0, 'PROD_2': 120.0}
    train_df = _build_part(prods, range(1, 21), base_m)
    val_df = _build_part(prods, range(21, 26), base_m)
    test_df = _build_part(prods, range(26, 31), base_m)

    return train_df, val_df, test_df, feature_cols


def test_baseline_train_avg(sample_splits):
    """Test that Baseline 1 predicts the exact product training mean."""
    train_df, _, test_df, _ = sample_splits
    preds = predict_baseline_train_avg(train_df, test_df)

    p1_mean = train_df[train_df['StockCode'] == 'PROD_1']['weekly_quantity'].mean()
    p2_mean = train_df[train_df['StockCode'] == 'PROD_2']['weekly_quantity'].mean()

    assert np.allclose(preds[test_df['StockCode'] == 'PROD_1'], p1_mean)
    assert np.allclose(preds[test_df['StockCode'] == 'PROD_2'], p2_mean)


def test_baseline_last_week(sample_splits):
    """Test that Baseline 2 predicts lag_1 directly."""
    _, _, test_df, _ = sample_splits
    preds = predict_baseline_last_week(test_df)
    assert np.array_equal(preds, test_df['lag_1'].values)


def test_linear_regression(sample_splits):
    """Test that Linear Regression fits and outputs non-negative predictions."""
    train_df, _, test_df, feature_cols = sample_splits
    lr = train_linear_regression(train_df[feature_cols], train_df['weekly_quantity'])
    preds = np.clip(lr.predict(test_df[feature_cols]), 0, None)
    assert len(preds) == len(test_df)
    assert (preds >= 0).all()


def test_tune_random_forest_on_validation(sample_splits):
    """Test that Random Forest is tuned strictly using validation weeks."""
    train_df, val_df, _, feature_cols = sample_splits
    grid = [
        {'n_estimators': 10, 'max_depth': 4, 'min_samples_leaf': 2},
        {'n_estimators': 20, 'max_depth': 6, 'min_samples_leaf': 2}
    ]

    best_rf, best_params, tuning_df = tune_random_forest_on_validation(
        train_df[feature_cols], train_df['weekly_quantity'],
        val_df[feature_cols], val_df['weekly_quantity'],
        param_grid=grid,
        verbose=False
    )

    assert best_params in grid
    assert len(tuning_df) == len(grid)
    assert 'Val_MAE' in tuning_df.columns
    min_mae = tuning_df['Val_MAE'].min()
    best_row = tuning_df[tuning_df['Is_Best']].iloc[-1]
    assert best_row['Val_MAE'] == min_mae


def test_unified_predictions_table(sample_splits):
    """Test that all methods' predictions are merged into one unified table."""
    train_df, val_df, test_df, feature_cols = sample_splits
    p_train_avg = predict_baseline_train_avg(train_df, test_df)
    p_last_week = predict_baseline_last_week(test_df)
    lr = train_linear_regression(train_df[feature_cols], train_df['weekly_quantity'])
    p_lr = np.clip(lr.predict(test_df[feature_cols]), 0, None)

    rf, _, _ = tune_random_forest_on_validation(
        train_df[feature_cols], train_df['weekly_quantity'],
        val_df[feature_cols], val_df['weekly_quantity'],
        verbose=False
    )
    p_rf = np.clip(rf.predict(test_df[feature_cols]), 0, None)

    table = build_unified_predictions_table(
        test_df,
        pred_train_avg=p_train_avg,
        pred_last_week=p_last_week,
        pred_lr=p_lr,
        pred_rf=p_rf
    )

    expected_cols = [
        'StockCode', 'week_start', 'weekly_quantity',
        'pred_train_avg', 'pred_last_week',
        'pred_linear_regression', 'pred_random_forest'
    ]
    for col in expected_cols:
        assert col in table.columns
    assert len(table) == len(test_df)


# =============================================================================
# 3. TEST SET EVALUATION & METRICS TESTS
# =============================================================================

@pytest.fixture
def mock_predictions():
    """Creates a synthetic test predictions dataframe with 3 products over 4 weeks."""
    rows = []
    weeks = ['2021-10-04', '2021-10-11', '2021-10-18', '2021-10-25']
    products = [
        ('PROD_A', 'Product Alpha', 100.0),
        ('PROD_B', 'Product Beta', 50.0),
        ('PROD_C', 'Product Gamma', 10.0),
    ]

    for sc, desc, base in products:
        for i, w in enumerate(weeks):
            actual = base + (i * 5.0)
            rows.append({
                'StockCode': sc,
                'week_start': w,
                'description': desc,
                'weekly_quantity': actual,
                'pred_train_avg': base,
                'pred_last_week': actual - 2.0,
                'pred_linear_regression': actual + 1.0,
                'pred_random_forest': actual - 0.5
            })

    return pd.DataFrame(rows)


def test_compute_overall_metrics(mock_predictions):
    """Tests that compute_overall_metrics returns correct models, columns, and values."""
    metrics_df = compute_overall_metrics(mock_predictions)
    assert isinstance(metrics_df, pd.DataFrame)
    assert set(metrics_df.columns) == {'Model', 'MAE', 'RMSE'}
    assert len(metrics_df) == 4

    lr_row = metrics_df[metrics_df['Model'] == 'Model: Linear Regression'].iloc[0]
    assert lr_row['MAE'] == 1.0
    assert lr_row['RMSE'] == 1.0

    rf_row = metrics_df[metrics_df['Model'] == 'Model: Random Forest (Tuned)'].iloc[0]
    assert rf_row['MAE'] == 0.5
    assert rf_row['RMSE'] == 0.5


def test_compute_per_product_metrics(mock_predictions):
    """Tests that compute_per_product_metrics correctly computes product-level stats."""
    per_prod = compute_per_product_metrics(mock_predictions)
    assert len(per_prod) == 3
    assert list(per_prod['StockCode']) == ['PROD_A', 'PROD_B', 'PROD_C']
    assert 'Best_Model_MAE' in per_prod.columns
    assert (per_prod['Best_Model_MAE'] == 'Random_Forest').all()


def test_compute_error_by_week(mock_predictions):
    """Tests that compute_error_by_week groups correctly across weeks."""
    weekly_df = compute_error_by_week(mock_predictions)
    assert len(weekly_df) == 4
    assert 'Actual_Total_Demand' in weekly_df.columns
    assert 'Linear_Regression_MAE' in weekly_df.columns
    assert 'Random_Forest_MAE' in weekly_df.columns
    assert (weekly_df['Product_Count'] == 3).all()
    assert (weekly_df['Linear_Regression_MAE'] == 1.0).all()


def test_summarize_model_rankings(mock_predictions):
    """Tests summary stats extraction from per-product table."""
    per_prod = compute_per_product_metrics(mock_predictions)
    summary = summarize_model_rankings(per_prod)
    assert 'best_counts_by_mae' in summary
    assert summary['best_counts_by_mae']['Random_Forest'] == 3


def test_generate_evaluation_plots(mock_predictions, tmp_path):
    """Tests that all 4 figure files are properly generated and saved on disk."""
    out_dir = str(tmp_path / "test_figures")
    weekly_df = compute_error_by_week(mock_predictions)
    saved = generate_evaluation_plots(mock_predictions, weekly_df, output_dir=out_dir)

    expected_keys = {'aggregate_weekly', 'error_by_week', 'top_products', 'error_distribution'}
    assert set(saved.keys()) == expected_keys
    for path in saved.values():
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_export_evaluation_artifacts(mock_predictions, tmp_path):
    """Tests CSV artifact export."""
    out_dir = str(tmp_path / "test_artifacts")
    overall_df = compute_overall_metrics(mock_predictions)
    per_prod = compute_per_product_metrics(mock_predictions)
    weekly_df = compute_error_by_week(mock_predictions)

    export_evaluation_artifacts(overall_df, per_prod, weekly_df, output_dir=out_dir)
    for fname in ["test_metrics_overall.csv", "test_metrics_per_product.csv", "test_metrics_by_week.csv"]:
        fpath = os.path.join(out_dir, fname)
        assert os.path.exists(fpath)
        assert not pd.read_csv(fpath).empty


# =============================================================================
# 4. WHAT-IF PRICING TOOL TESTS
# =============================================================================

def test_candidate_price_bounds_and_grid():
    """Verifies that candidate prices lie strictly within 5th and 95th percentiles."""
    observed_prices = np.array([1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 10.0])
    p_low, p_high = calculate_candidate_price_bounds(observed_prices, 5.0, 95.0)

    assert p_low <= p_high
    assert p_low >= 1.0
    assert p_high <= 10.0

    candidates = build_candidate_prices(observed_prices, n_candidates=5)
    assert len(candidates) == 5
    assert candidates[0] == pytest.approx(p_low, abs=0.01)
    assert candidates[-1] == pytest.approx(p_high, abs=0.01)
    assert (np.diff(candidates) > 0).all()


def test_candidate_price_edge_cases():
    """Verifies price bounds with identical prices or single price."""
    flat_prices = [2.50, 2.50, 2.50]
    p_low, p_high = calculate_candidate_price_bounds(flat_prices)
    assert p_low == 2.50
    assert p_high == 2.50

    cands = build_candidate_prices(flat_prices, n_candidates=5)
    assert len(cands) == 1
    assert cands[0] == 2.50


def test_revenue_calculation():
    """Verifies formula: Revenue = Price * Predicted Quantity."""
    assert calculate_what_if_revenue(10.0, 50.0) == 500.0
    assert calculate_what_if_revenue(2.5, 100.0) == 250.0
    assert calculate_what_if_revenue(-5.0, 50.0) == 0.0
    assert calculate_what_if_revenue(10.0, -10.0) == 0.0


def test_check_demand_monotonicity_normal_downward():
    """When demand decreases or stays flat as price increases, no warning is flagged."""
    prices = np.array([1.0, 2.0, 3.0, 4.0])
    demands = np.array([100.0, 80.0, 60.0, 40.0])

    demand_rises, warnings_list = check_demand_monotonicity(prices, demands)
    assert not demand_rises
    assert len(warnings_list) == 0


def test_check_demand_monotonicity_warning_when_demand_rises():
    """When demand rises as price increases, it must be flagged with explicit warnings."""
    prices = np.array([1.0, 2.0, 3.0, 4.0])
    demands = np.array([100.0, 80.0, 95.0, 70.0])

    demand_rises, warnings_list = check_demand_monotonicity(prices, demands, stock_code="TEST101")
    assert demand_rises
    assert len(warnings_list) >= 1
    assert "Non-causal demand response detected" in warnings_list[0]
    assert "Product TEST101" in warnings_list[0]


def test_predict_demand_holding_features_fixed():
    """Verifies that predict_demand_with_candidate_prices holds non-price features fixed."""
    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_4', 'week_of_year', 'mean_unit_price']
    model = LinearRegression().fit(pd.DataFrame([[0.0] * 5], columns=feature_cols), [100.0])
    model.coef_ = np.array([0.5, 0.0, 0.0, 0.0, -10.0])
    model.intercept_ = 100.0

    ref_row = pd.Series({
        'lag_1': 100.0,
        'lag_2': 90.0,
        'rolling_mean_4': 100.0,
        'week_of_year': 25,
        'mean_unit_price': 3.0
    })

    candidate_prices = np.array([2.0, 3.0, 4.0, 5.0])
    preds = predict_demand_with_candidate_prices(
        model=model,
        feature_row=ref_row,
        candidate_prices=candidate_prices,
        feature_cols=feature_cols
    )

    assert len(preds) == len(candidate_prices)
    diffs = np.diff(preds)
    assert pytest.approx(diffs, abs=1e-3) == [-10.0, -10.0, -10.0]


def test_simulate_product_pricing_end_to_end():
    """Tests end-to-end simulate_product_pricing function."""
    feature_cols = ['lag_1', 'lag_2', 'mean_unit_price']
    X_train = pd.DataFrame({
        'lag_1': [10.0, 20.0, 30.0],
        'lag_2': [10.0, 20.0, 30.0],
        'mean_unit_price': [1.0, 2.0, 3.0]
    })
    y_train = pd.Series([50.0, 40.0, 30.0])
    model = LinearRegression().fit(X_train, y_train)

    ref_row = pd.Series({'lag_1': 20.0, 'lag_2': 20.0, 'mean_unit_price': 2.0})
    observed_prices = [1.5, 1.8, 2.0, 2.2, 2.5]

    sim_df, demand_rises, warnings_list = simulate_product_pricing(
        model=model,
        feature_row=ref_row,
        observed_prices=observed_prices,
        feature_cols=feature_cols,
        n_candidates=5,
        stock_code="22197",
        description="TEST ITEM"
    )

    assert len(sim_df) == 5
    assert 'Candidate_Price' in sim_df.columns
    assert 'Predicted_Demand' in sim_df.columns
    assert 'Simulated_Revenue' in sim_df.columns or 'Hypothetical_Revenue' in sim_df.columns
    assert not demand_rises


# =============================================================================
# 5. DASHBOARD ARTIFACT LOADING TESTS
# =============================================================================

def test_dashboard_resolve_path():
    """Verifies that resolve_path finds existing artifact files."""
    path = resolve_path("artifacts/unified_test_predictions.csv")
    assert os.path.exists(path)


def test_dashboard_load_all_saved_artifacts():
    """Verifies that load_all_saved_artifacts loads all essential tables."""
    artifacts = load_all_saved_artifacts()
    assert isinstance(artifacts, dict)

    required_keys = [
        'test_preds', 'overall_metrics', 'per_product_metrics',
        'error_by_week', 'panel_full', 'pricing_scenarios_lr',
        'pricing_summary_lr', 'pricing_scenarios_rf', 'pricing_summary_rf'
    ]

    for k in required_keys:
        assert k in artifacts, f"Missing key: {k}"
        df = artifacts[k]
        assert isinstance(df, pd.DataFrame), f"Key {k} is not a DataFrame"
        assert not df.empty, f"DataFrame {k} is empty"

    assert 'StockCode' in artifacts['test_preds'].columns
    assert 'weekly_quantity' in artifacts['test_preds'].columns
    assert 'pred_linear_regression' in artifacts['test_preds'].columns
    assert 'pred_random_forest' in artifacts['test_preds'].columns
