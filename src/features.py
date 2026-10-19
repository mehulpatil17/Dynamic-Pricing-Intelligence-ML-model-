"""
src/features.py
Module for building product-week demand panels, engineering strictly causal
time-series features (no lookahead leakage), and performing chronological splits.

Key Functions:
- build_product_week_table: Aggregates by product and week start, sums Quantity,
  averages UnitPrice, and fills missing product-weeks with 0 units.
- engineer_features: Generates lag_1, lag_2, lag_4, shifted trailing rolling mean,
  and calendar week_of_year.
- split_chronological_70_15_15: Splits weeks chronologically into 70% Train,
  15% Validation, and 15% Test.
"""

from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np


def build_product_week_table(
    df: pd.DataFrame,
    product_col: str = 'StockCode',
    date_col: str = 'InvoiceDate',
    quantity_col: str = 'Quantity',
    price_col: str = 'UnitPrice',
    description_col: str = 'Description',
    top_n_products: Optional[int] = None,
    min_active_weeks: Optional[int] = None
) -> pd.DataFrame:
    """
    Builds a continuous, balanced product-week demand table.
    
    For every product:
    - Sums Quantity per calendar week (net weekly demand, clipped to >= 0).
    - Computes mean UnitPrice in weeks with sales.
    - Explicitly fills missing calendar weeks with 0 units.
    - Imputes missing week UnitPrice using forward fill, backward fill, and product mean.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction records.
    product_col : str
        Product identifier column ('StockCode').
    date_col : str
        Transaction timestamp column ('InvoiceDate').
    quantity_col : str
        Sales quantity column ('Quantity').
    price_col : str
        Unit price column ('UnitPrice').
    description_col : str
        Product description column ('Description').
    top_n_products : Optional[int]
        If set, filters to the top N products by total sales volume.
    min_active_weeks : Optional[int]
        If set, filters to products active in at least this many weeks.
        
    Returns
    -------
    pd.DataFrame
        Balanced product-week panel sorted by [StockCode, week_start].
    """
    df_work = df.copy()

    # Ensure week_start is created as Monday start time
    if 'week_start' not in df_work.columns:
        if 'Week_Start' in df_work.columns:
            df_work['week_start'] = pd.to_datetime(df_work['Week_Start'])
        else:
            df_work['week_start'] = pd.to_datetime(df_work[date_col]).dt.to_period('W-SUN').dt.start_time

    df_work[product_col] = df_work[product_col].astype(str).str.strip()

    # 1. Aggregate by product and week
    weekly_agg = df_work.groupby([product_col, 'week_start']).agg(
        weekly_quantity=(quantity_col, 'sum'),
        mean_unit_price=(price_col, 'mean'),
        description=(description_col, 'first')
    ).reset_index()

    # Net weekly demand cannot be negative in retail inventory
    weekly_agg['weekly_quantity'] = weekly_agg['weekly_quantity'].clip(lower=0.0)

    # 2. Product-level metadata
    prod_meta = weekly_agg.groupby(product_col).agg(
        description=('description', 'first'),
        overall_mean_price=('mean_unit_price', 'mean'),
        total_volume=('weekly_quantity', 'sum'),
        active_weeks=('week_start', 'nunique')
    ).reset_index()

    # Optional product filtering
    if top_n_products is not None:
        selected_prods = prod_meta.nlargest(top_n_products, 'total_volume')[product_col].values
    elif min_active_weeks is not None:
        selected_prods = prod_meta[prod_meta['active_weeks'] >= min_active_weeks][product_col].values
    else:
        selected_prods = prod_meta[product_col].values

    weekly_sub = weekly_agg[weekly_agg[product_col].isin(selected_prods)].copy()
    prod_meta_sub = prod_meta[prod_meta[product_col].isin(selected_prods)].copy()

    # 3. Create full continuous grid of all calendar weeks
    min_week = weekly_sub['week_start'].min()
    max_week = weekly_sub['week_start'].max()
    all_weeks = pd.date_range(start=min_week, end=max_week, freq='W-MON')

    full_grid_idx = pd.MultiIndex.from_product(
        [selected_prods, all_weeks],
        names=[product_col, 'week_start']
    )

    panel = weekly_sub.set_index([product_col, 'week_start']).reindex(full_grid_idx).reset_index()

    # 4. Fill missing weeks with 0 units
    panel['weekly_quantity'] = panel['weekly_quantity'].fillna(0.0)

    # Merge product metadata
    panel = panel.drop(columns=['description'], errors='ignore')
    panel = panel.merge(
        prod_meta_sub[[product_col, 'description', 'overall_mean_price']],
        on=product_col,
        how='left'
    )

    # Impute UnitPrice for zero-sales weeks: forward fill -> backward fill -> overall mean
    panel = panel.sort_values([product_col, 'week_start']).reset_index(drop=True)
    panel['mean_unit_price'] = (
        panel.groupby(product_col)['mean_unit_price']
        .ffill()
        .bfill()
        .fillna(panel['overall_mean_price'])
    )

    # Standardize column naming and sorting
    panel['description'] = panel['description'].fillna("")
    return panel.sort_values([product_col, 'week_start']).reset_index(drop=True)


def engineer_features(
    panel_df: pd.DataFrame,
    product_col: str = 'StockCode',
    date_col: str = 'week_start',
    target_col: str = 'weekly_quantity',
    price_col: str = 'mean_unit_price',
    lags: Tuple[int, ...] = (1, 2, 4),
    rolling_window: int = 4
) -> pd.DataFrame:
    """
    Engineers causal time-series features strictly avoiding future lookahead leakage.
    
    Features engineered:
    - lag_1: Demand at week t-1
    - lag_2: Demand at week t-2
    - lag_4: Demand at week t-4
    - rolling_mean_4: Trailing moving average over past 4 weeks (shifted by 1 so
      week t is strictly excluded from its own moving average)
    - week_of_year: Integer week of the year (1-52) from week_start
    - mean_unit_price: Unit selling price
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Balanced product-week panel.
    product_col : str
        Product identifier column.
    date_col : str
        Week start date column.
    target_col : str
        Target demand column.
    price_col : str
        Price column.
    lags : Tuple[int, ...]
        Lags to compute (default: 1, 2, 4).
    rolling_window : int
        Window length for trailing rolling mean (default: 4).
        
    Returns
    -------
    pd.DataFrame
        Dataframe containing features with initial warm-up weeks dropped.
    """
    df_feat = panel_df.copy()
    df_feat = df_feat.sort_values([product_col, date_col]).reset_index(drop=True)

    # 1. Compute Lags
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat.groupby(product_col)[target_col].shift(lag)

    # 2. Shifted Rolling Mean (Shift by 1 ensures strictly historical weeks are used)
    # E.g. at week t, rolling_mean_4 = mean(quantity[t-1], quantity[t-2], quantity[t-3], quantity[t-4])
    shifted_qty = df_feat.groupby(product_col)[target_col].shift(1)
    df_feat[f'rolling_mean_{rolling_window}'] = (
        shifted_qty.groupby(df_feat[product_col])
        .rolling(rolling_window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Trailing rolling standard deviation
    df_feat[f'rolling_std_{rolling_window}'] = (
        shifted_qty.groupby(df_feat[product_col])
        .rolling(rolling_window)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # 3. Calendar Feature: Week of Year
    df_feat['week_of_year'] = pd.to_datetime(df_feat[date_col]).dt.isocalendar().week.astype(int)

    # 4. Drop warm-up rows where required lag or rolling features are NaN
    max_lag = max(max(lags), rolling_window)
    required_cols = [f'lag_{lag}' for lag in lags] + [f'rolling_mean_{rolling_window}']
    df_feat = df_feat.dropna(subset=required_cols).reset_index(drop=True)

    return df_feat


def split_chronological_70_15_15(
    df: pd.DataFrame,
    date_col: str = 'week_start',
    target_col: str = 'weekly_quantity'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str], str]:
    """
    Splits the continuous timeline chronologically into 70% Train, 15% Validation, and 15% Test.
    
    Guarantees strict chronological ordering with zero temporal overlap:
        max(train_weeks) < min(val_weeks) <= max(val_weeks) < min(test_weeks)
        
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing engineered features.
    date_col : str
        Week start timestamp column.
    target_col : str
        Demand target column.
        
    Returns
    -------
    Tuple containing:
    - train_df : pd.DataFrame (70% of chronological weeks)
    - val_df : pd.DataFrame (15% of chronological weeks)
    - test_df : pd.DataFrame (15% of chronological weeks)
    - split_info : Dict[str, Any] (week ranges, split dates, row counts)
    - feature_cols : List[str] (feature column names)
    - target_col : str ('weekly_quantity')
    """
    unique_weeks = sorted(df[date_col].unique())
    total_weeks = len(unique_weeks)

    if total_weeks < 10:
        raise ValueError(f"Need at least 10 weeks to split 70/15/15; got {total_weeks}.")

    # Calculate week counts
    n_train_weeks = int(round(total_weeks * 0.70))
    n_val_weeks = int(round(total_weeks * 0.15))
    n_test_weeks = total_weeks - n_train_weeks - n_val_weeks

    # Partition week lists
    train_weeks = unique_weeks[:n_train_weeks]
    val_weeks = unique_weeks[n_train_weeks : n_train_weeks + n_val_weeks]
    test_weeks = unique_weeks[n_train_weeks + n_val_weeks :]

    val_split_date = val_weeks[0]
    test_split_date = test_weeks[0]

    # Subset dataframes
    train_df = df[df[date_col].isin(train_weeks)].copy().reset_index(drop=True)
    val_df = df[df[date_col].isin(val_weeks)].copy().reset_index(drop=True)
    test_df = df[df[date_col].isin(test_weeks)].copy().reset_index(drop=True)

    feature_cols = [
        'lag_1', 'lag_2', 'lag_4',
        'rolling_mean_4', 'rolling_std_4',
        'week_of_year',
        'mean_unit_price'
    ]

    split_info = {
        'total_weeks': total_weeks,
        'train_weeks_count': len(train_weeks),
        'val_weeks_count': len(val_weeks),
        'test_weeks_count': len(test_weeks),
        'train_date_range': (train_weeks[0], train_weeks[-1]),
        'val_date_range': (val_weeks[0], val_weeks[-1]),
        'test_date_range': (test_weeks[0], test_weeks[-1]),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'val_split_date': val_split_date,
        'test_split_date': test_split_date
    }

    return train_df, val_df, test_df, split_info, feature_cols, target_col
