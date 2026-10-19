"""
tests/test_features.py
Pytest test suite proving:
1. build_product_week_table fills missing weeks with 0 units.
2. NO feature uses the current week (t) or any later week (t+1, t+2, ...).
3. Exact mathematical alignment of lag_1, lag_2, lag_4, and shifted rolling_mean_4.
4. Strict chronological ordering and zero overlap of 70/15/15 split.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import (
    build_product_week_table,
    engineer_features,
    split_chronological_70_15_15
)
from src.cleaning import (
    is_non_product_code,
    audit_raw_data,
    clean_online_retail_data
)


@pytest.fixture
def sample_synthetic_transactions():
    """
    Creates a controlled synthetic transaction dataset spanning 10 consecutive weeks.
    Product A has sales in weeks 1, 2, 5, 8, 10 (with gaps in weeks 3, 4, 6, 7, 9).
    Product B has sales in weeks 2, 3, 4, 5, 6, 7, 8, 9, 10.
    """
    weeks = pd.date_range(start='2021-01-04', periods=10, freq='W-MON')
    records = []

    # Product A: Sporadic sales with gaps
    for idx in [0, 1, 4, 7, 9]:
        records.append({
            'StockCode': 'PROD_A',
            'InvoiceDate': weeks[idx] + pd.Timedelta(days=1),
            'Quantity': (idx + 1) * 10,
            'UnitPrice': 5.00,
            'Description': 'Item A'
        })

    # Product B: Regular sales
    for idx in range(1, 10):
        records.append({
            'StockCode': 'PROD_B',
            'InvoiceDate': weeks[idx] + pd.Timedelta(days=2),
            'Quantity': 20,
            'UnitPrice': 12.50,
            'Description': 'Item B'
        })

    return pd.DataFrame(records)


def test_build_product_week_table_zero_filling(sample_synthetic_transactions):
    """
    Test that build_product_week_table creates a balanced continuous panel
    and correctly fills missing weeks with 0 units.
    """
    df = sample_synthetic_transactions
    panel = build_product_week_table(df)

    # 1. Total weeks should be exactly 10 for each product
    prod_a = panel[panel['StockCode'] == 'PROD_A'].sort_values('week_start').reset_index(drop=True)
    assert len(prod_a) == 10, f"Expected 10 weeks for PROD_A, got {len(prod_a)}"

    # 2. Check that missing weeks (indices 2, 3 -> weeks 3 and 4) have exactly 0 quantity
    assert prod_a.loc[2, 'weekly_quantity'] == 0.0
    assert prod_a.loc[3, 'weekly_quantity'] == 0.0
    assert prod_a.loc[5, 'weekly_quantity'] == 0.0
    assert prod_a.loc[6, 'weekly_quantity'] == 0.0
    assert prod_a.loc[8, 'weekly_quantity'] == 0.0

    # 3. Check that active weeks retain their exact demand sum
    assert prod_a.loc[0, 'weekly_quantity'] == 10.0
    assert prod_a.loc[1, 'weekly_quantity'] == 20.0
    assert prod_a.loc[4, 'weekly_quantity'] == 50.0

    # 4. Check price imputation: missing weeks should have a valid price (non-null, non-zero)
    assert not prod_a['mean_unit_price'].isnull().any()
    assert (prod_a['mean_unit_price'] == 5.00).all()


def test_no_lookahead_leakage_current_week():
    """
    CRUCIAL TEST: Proves that NO feature at week t depends on week t quantity.
    
    If we modify the target quantity at week t (e.g. from 50 to 999,999),
    all features engineered for week t MUST REMAIN COMPLETELY UNCHANGED.
    """
    weeks = pd.date_range(start='2021-01-04', periods=8, freq='W-MON')
    quantities = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    panel_original = pd.DataFrame({
        'StockCode': 'PROD_X',
        'week_start': weeks,
        'weekly_quantity': quantities,
        'mean_unit_price': 10.0,
        'description': 'Item X'
    })

    # Engineer features on original data
    feat_orig = engineer_features(panel_original, lags=(1, 2, 4), rolling_window=4)

    # Pick test evaluation row at week index 5 (target was 60.0)
    target_date = weeks[5]
    orig_row_t = feat_orig[feat_orig['week_start'] == target_date].iloc[0]

    # Now create perturbed dataset where week t's quantity is drastically changed (+1,000,000 units)
    panel_perturbed = panel_original.copy()
    panel_perturbed.loc[panel_perturbed['week_start'] == target_date, 'weekly_quantity'] = 1_000_000.0

    # Re-engineer features on perturbed data
    feat_perturbed = engineer_features(panel_perturbed, lags=(1, 2, 4), rolling_window=4)
    perturbed_row_t = feat_perturbed[feat_perturbed['week_start'] == target_date].iloc[0]

    # ASSERTION: Features at week t must be 100% IDENTICAL despite massive change in week t quantity
    assert orig_row_t['lag_1'] == perturbed_row_t['lag_1'], "lag_1 leaked current week data!"
    assert orig_row_t['lag_2'] == perturbed_row_t['lag_2'], "lag_2 leaked current week data!"
    assert orig_row_t['lag_4'] == perturbed_row_t['lag_4'], "lag_4 leaked current week data!"
    assert orig_row_t['rolling_mean_4'] == perturbed_row_t['rolling_mean_4'], "rolling_mean_4 leaked current week data!"
    assert orig_row_t['rolling_std_4'] == perturbed_row_t['rolling_std_4'], "rolling_std_4 leaked current week data!"
    assert orig_row_t['week_of_year'] == perturbed_row_t['week_of_year'], "week_of_year altered!"


def test_no_lookahead_leakage_future_weeks():
    """
    CRUCIAL TEST: Proves that NO feature at week t depends on future weeks (t+1, t+2, ...).
    
    If future weeks are corrupted or randomized, the feature values at week t
    and all earlier weeks must remain strictly invariant.
    """
    weeks = pd.date_range(start='2021-01-04', periods=10, freq='W-MON')
    panel_orig = pd.DataFrame({
        'StockCode': 'PROD_Y',
        'week_start': weeks,
        'weekly_quantity': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
        'mean_unit_price': 3.50,
        'description': 'Item Y'
    })

    feat_orig = engineer_features(panel_orig, lags=(1, 2, 4), rolling_window=4)

    # Evaluate at week index 6
    eval_date = weeks[6]
    features_before = feat_orig[feat_orig['week_start'] <= eval_date][
        ['lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4']
    ].copy()

    # Corrupt future weeks (indices 7, 8, 9)
    panel_future_corrupted = panel_orig.copy()
    panel_future_corrupted.loc[panel_future_corrupted['week_start'] > eval_date, 'weekly_quantity'] = [9999.0, -500.0, 88888.0]

    feat_after = engineer_features(panel_future_corrupted, lags=(1, 2, 4), rolling_window=4)
    features_after = feat_after[feat_after['week_start'] <= eval_date][
        ['lag_1', 'lag_2', 'lag_4', 'rolling_mean_4', 'rolling_std_4']
    ].copy()

    # ASSERTION: All features for week t and all prior weeks must match exactly
    pd.testing.assert_frame_equal(features_before, features_after)


def test_lag_and_rolling_exact_mathematical_values():
    """
    Tests exact mathematical alignment of engineered features:
    lag_1, lag_2, lag_4, and rolling_mean_4.
    """
    weeks = pd.date_range(start='2021-01-04', periods=7, freq='W-MON')
    # Known sequence: [10, 20, 30, 40, 50, 60, 70]
    # Indices:         0   1   2   3   4   5   6
    raw_quantities = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    panel = pd.DataFrame({
        'StockCode': 'PROD_M',
        'week_start': weeks,
        'weekly_quantity': raw_quantities,
        'mean_unit_price': 1.0,
        'description': 'Math Test Item'
    })

    feat_df = engineer_features(panel, lags=(1, 2, 4), rolling_window=4)

    # For week index 4 (week_start = weeks[4], target = 50.0):
    # lag_1 should be 40.0 (index 3)
    # lag_2 should be 30.0 (index 2)
    # lag_4 should be 10.0 (index 0)
    # rolling_mean_4 should be mean(40, 30, 20, 10) = 25.0
    row_4 = feat_df[feat_df['week_start'] == weeks[4]].iloc[0]
    assert row_4['lag_1'] == 40.0
    assert row_4['lag_2'] == 30.0
    assert row_4['lag_4'] == 10.0
    assert row_4['rolling_mean_4'] == 25.0

    # For week index 6 (week_start = weeks[6], target = 70.0):
    # lag_1 should be 60.0 (index 5)
    # lag_2 should be 50.0 (index 4)
    # lag_4 should be 30.0 (index 2)
    # rolling_mean_4 should be mean(60, 50, 40, 30) = 45.0
    row_6 = feat_df[feat_df['week_start'] == weeks[6]].iloc[0]
    assert row_6['lag_1'] == 60.0
    assert row_6['lag_2'] == 50.0
    assert row_6['lag_4'] == 30.0
    assert row_6['rolling_mean_4'] == 45.0


def test_chronological_70_15_15_split_boundaries():
    """
    Tests strict chronological 70/15/15 split:
    1. Zero temporal overlap between Train, Validation, and Test.
    2. max(train_date) < min(val_date)
    3. max(val_date) < min(test_date)
    4. Week partition counts sum to total weeks.
    """
    # 20 distinct consecutive weeks
    weeks = pd.date_range(start='2021-01-04', periods=20, freq='W-MON')
    df = pd.DataFrame({
        'StockCode': 'PROD_SPLIT',
        'week_start': weeks,
        'weekly_quantity': [10.0] * 20,
        'lag_1': [10.0] * 20,
        'lag_2': [10.0] * 20,
        'lag_4': [10.0] * 20,
        'rolling_mean_4': [10.0] * 20,
        'rolling_std_4': [0.0] * 20,
        'week_of_year': [w.week for w in weeks],
        'mean_unit_price': [2.5] * 20
    })

    train_df, val_df, test_df, split_info, feature_cols, target_col = split_chronological_70_15_15(df)

    # 1. Total weeks = 20 -> 70% = 14 weeks, 15% = 3 weeks, 15% = 3 weeks
    assert split_info['total_weeks'] == 20
    assert split_info['train_weeks_count'] == 14
    assert split_info['val_weeks_count'] == 3
    assert split_info['test_weeks_count'] == 3
    assert len(train_df) + len(val_df) + len(test_df) == 20

    # 2. Strict chronological ordering (no overlap)
    train_max = train_df['week_start'].max()
    val_min = val_df['week_start'].min()
    val_max = val_df['week_start'].max()
    test_min = test_df['week_start'].min()

    assert train_max < val_min, f"Train max ({train_max}) must be strictly less than Val min ({val_min})"
    assert val_max < test_min, f"Val max ({val_max}) must be strictly less than Test min ({test_min})"

    # 3. Disjoint set of weeks
    train_w_set = set(train_df['week_start'])
    val_w_set = set(val_df['week_start'])
    test_w_set = set(test_df['week_start'])

    assert len(train_w_set.intersection(val_w_set)) == 0, "Train and Val weeks overlap!"
    assert len(val_w_set.intersection(test_w_set)) == 0, "Val and Test weeks overlap!"
    assert len(train_w_set.intersection(test_w_set)) == 0, "Train and Test weeks overlap!"


# -----------------------------------------------------------------------------
# Data Audit & Cleaning Pipeline Tests
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_raw_data():
    """
    Creates a controlled synthetic raw transaction dataset containing:
    - 2 valid product transactions
    - 1 duplicate row
    - 1 cancellation ('C' prefix)
    - 1 negative quantity (return)
    - 1 zero unit price (giveaway/free sample)
    - 2 non-product service codes ('POST', 'M')
    """
    rows = [
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HEART T-LIGHT", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "UnitPrice": 2.55, "CustomerID": 17850, "Country": "United Kingdom"},
        {"InvoiceNo": "536365", "StockCode": "71053", "Description": "WHITE METAL LANTERN", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "UnitPrice": 3.39, "CustomerID": 17850, "Country": "United Kingdom"},
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HEART T-LIGHT", "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "UnitPrice": 2.55, "CustomerID": 17850, "Country": "United Kingdom"},
        {"InvoiceNo": "C536379", "StockCode": "D", "Description": "Discount", "Quantity": -1, "InvoiceDate": "2010-12-01 09:41:00", "UnitPrice": 27.50, "CustomerID": 14527, "Country": "United Kingdom"},
        {"InvoiceNo": "536380", "StockCode": "22960", "Description": "JAM MAKING SET", "Quantity": -5, "InvoiceDate": "2010-12-01 09:45:00", "UnitPrice": 4.25, "CustomerID": 15311, "Country": "United Kingdom"},
        {"InvoiceNo": "536381", "StockCode": "21730", "Description": "GLASS STAR FROSTED", "Quantity": 10, "InvoiceDate": "2010-12-01 09:49:00", "UnitPrice": 0.00, "CustomerID": 15311, "Country": "United Kingdom"},
        {"InvoiceNo": "536382", "StockCode": "POST", "Description": "POSTAGE", "Quantity": 1, "InvoiceDate": "2010-12-01 09:55:00", "UnitPrice": 18.00, "CustomerID": 15311, "Country": "United Kingdom"},
        {"InvoiceNo": "536383", "StockCode": "M", "Description": "Manual", "Quantity": 2, "InvoiceDate": "2010-12-01 10:00:00", "UnitPrice": 5.00, "CustomerID": 15311, "Country": "United Kingdom"}
    ]
    return pd.DataFrame(rows)


def test_is_non_product_code():
    """
    Verifies that known non-merchandise service codes and gift vouchers are detected.
    """
    test_codes = pd.Series(['85099B', '22197', 'POST', 'DOT', 'M', 'D', 'C2', 'CRUK', 'gift_0001_10', '15036'])
    is_non_prod = is_non_product_code(test_codes)

    assert not is_non_prod[0]  # 85099B
    assert not is_non_prod[1]  # 22197
    assert not is_non_prod[9]  # 15036
    assert is_non_prod[2]  # POST
    assert is_non_prod[3]  # DOT
    assert is_non_prod[4]  # M
    assert is_non_prod[5]  # D
    assert is_non_prod[6]  # C2
    assert is_non_prod[7]  # CRUK
    assert is_non_prod[8]  # gift_0001_10


def test_audit_raw_data(sample_raw_data):
    """
    Verifies initial audit statistics computation on raw dataset.
    """
    audit = audit_raw_data(sample_raw_data)
    assert audit['total_rows'] == 8
    assert audit['duplicates_count'] == 1
    assert audit['cancellations_count'] == 1
    assert audit['non_pos_qty_count'] == 2
    assert audit['non_pos_price_count'] == 1
    assert audit['non_product_codes_count'] == 3


def test_clean_online_retail_data_sequential_removal(sample_raw_data):
    """
    Verifies the sequential cleaning pipeline and exact removal logging.
    """
    clean_df, removal_logs = clean_online_retail_data(sample_raw_data, verbose=False)
    assert len(clean_df) == 2
    assert len(removal_logs) == 5

    r1 = removal_logs[removal_logs['Rule'].str.contains('duplicate', case=False)].iloc[0]
    assert r1['Rows_Removed'] == 1

    r2 = removal_logs[removal_logs['Rule'].str.contains('cancellation', case=False)].iloc[0]
    assert r2['Rows_Removed'] == 1

    r3 = removal_logs[removal_logs['Rule'].str.contains('quantity', case=False)].iloc[0]
    assert r3['Rows_Removed'] == 1

    r4 = removal_logs[removal_logs['Rule'].str.contains('unitprice', case=False)].iloc[0]
    assert r4['Rows_Removed'] == 1

    r5 = removal_logs[removal_logs['Rule'].str.contains('non-product', case=False)].iloc[0]
    assert r5['Rows_Removed'] == 2


def test_cleaned_dataset_invariants(sample_raw_data):
    """
    Verifies strict output invariants on the cleaned dataframe:
    - Zero duplicates
    - Zero 'C' cancellations
    - All Quantity > 0
    - All UnitPrice > 0
    - No non-product service codes
    """
    clean_df, _ = clean_online_retail_data(sample_raw_data, verbose=False)

    assert not clean_df.duplicated().any()
    assert not clean_df['InvoiceNo'].astype(str).str.startswith('C').any()
    assert (clean_df['Quantity'] > 0).all()
    assert (clean_df['UnitPrice'] > 0).all()
    assert not is_non_product_code(clean_df['StockCode']).any()

