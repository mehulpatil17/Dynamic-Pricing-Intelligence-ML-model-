"""
src/cleaning.py
Module for loading, auditing, and cleaning the UCI Online Retail dataset.

Performs:
1. Initial audit: Missing values, duplicates, cancellations, non-positive values, non-product StockCodes.
2. Step-by-step cleaning rules with exact row removal counts printed.
3. Exports cleaned dataset for downstream time-series panel creation and modeling.
"""

import os
import time
from typing import Dict, Any, Tuple, Set
import pandas as pd


# Comprehensive set of non-merchandise service, fee, and administrative codes in UCI Online Retail
NON_PRODUCT_CODES: Set[str] = {
    'POST',         # Postage fees
    'DOT',          # Dotcom postage
    'M',            # Manual adjustment
    'm',            # Manual adjustment (lowercase)
    'C2',           # Carriage
    'D',            # Discount
    'S',            # Samples
    'BANK CHARGES', # Bank transaction fees
    'AMAZONFEE',    # Amazon administrative fees
    'CRUK',         # CRUK charity commission
    'PADS',         # Cushion pads adjustment
    'B'             # Bad debt adjustment
}


def is_non_product_code(stock_code_series: pd.Series) -> pd.Series:
    """
    Identifies non-merchandise service codes or gift vouchers.
    
    Parameters
    ----------
    stock_code_series : pd.Series
        Series of StockCode values.
        
    Returns
    -------
    pd.Series (bool)
        Boolean mask where True indicates a non-product code.
    """
    clean_codes = stock_code_series.astype(str).str.strip()
    is_service = clean_codes.isin(NON_PRODUCT_CODES)
    is_gift_voucher = clean_codes.str.startswith('gift_')
    return is_service | is_gift_voucher


def audit_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs a comprehensive initial audit on raw dataset rows before any cleaning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as loaded from Online Retail.xlsx.
        
    Returns
    -------
    Dict[str, Any]
        Audit statistics dictionary.
    """
    total_rows = len(df)
    missing_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated().sum())
    cancellations_count = int(df['InvoiceNo'].astype(str).str.startswith('C').sum())
    non_pos_qty_count = int((df['Quantity'] <= 0).sum())
    non_pos_price_count = int((df['UnitPrice'] <= 0).sum())
    non_prod_count = int(is_non_product_code(df['StockCode']).sum())

    return {
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "duplicates_count": duplicates_count,
        "cancellations_count": cancellations_count,
        "non_pos_qty_count": non_pos_qty_count,
        "non_pos_price_count": non_pos_price_count,
        "non_product_codes_count": non_prod_count
    }


def print_audit_report(audit_stats: Dict[str, Any]) -> None:
    """
    Prints a formatted initial audit report.
    """
    total = audit_stats["total_rows"]
    print("=" * 75)
    print("1. INITIAL DATA AUDIT (RAW DATASET: Online Retail.xlsx)")
    print("=" * 75)
    print(f"Total raw rows: {total:,} | Columns: 8\n")

    print("• Missing Values per Column:")
    for col, count in audit_stats["missing_counts"].items():
        pct = (count / total) * 100
        print(f"    - {col:15s}: {count:7,d} missing ({pct:5.2f}%)")

    print(f"\n• Duplicates:")
    n_dup = audit_stats['duplicates_count']
    print(f"    - Exact duplicate rows: {n_dup:,d} ({(n_dup/total)*100:.2f}%)")

    print(f"\n• Cancellations:")
    n_can = audit_stats['cancellations_count']
    print(f"    - Invoices starting with 'C': {n_can:,d} ({(n_can/total)*100:.2f}%)")

    print(f"\n• Non-Positive Values:")
    n_q = audit_stats['non_pos_qty_count']
    n_p = audit_stats['non_pos_price_count']
    print(f"    - Quantity <= 0: {n_q:,d} ({(n_q/total)*100:.2f}%)")
    print(f"    - UnitPrice <= 0: {n_p:,d} ({(n_p/total)*100:.2f}%)")

    print(f"\n• Non-Product StockCodes:")
    n_np = audit_stats['non_product_codes_count']
    print(f"    - Service/Fee/Voucher codes: {n_np:,d} ({(n_np/total)*100:.2f}%)")
    print("=" * 75)


def clean_online_retail_data(
    df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies sequential cleaning rules and logs row counts removed at each step.
    
    Cleaning Rules (Applied Sequentially):
    1. Remove exact duplicate rows.
    2. Remove cancellations (InvoiceNo starting with 'C').
    3. Remove non-positive Quantity (Quantity <= 0).
    4. Remove non-positive UnitPrice (UnitPrice <= 0).
    5. Remove non-product StockCodes (postage, bank charges, discounts, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    verbose : bool
        Whether to print row counts removed at each step.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        cleaned_df, removal_log_df
    """
    current_df = df.copy()
    initial_count = len(current_df)
    removal_logs = []

    if verbose:
        print("\n" + "=" * 75)
        print("2. APPLYING CLEANING RULES (SEQUENTIAL REMOVAL COUNTS)")
        print("=" * 75)

    # Rule 1: Drop Duplicates
    count_before = len(current_df)
    current_df = current_df.drop_duplicates()
    removed_r1 = count_before - len(current_df)
    removal_logs.append({
        "Rule": "Rule 1: Remove Exact Duplicates",
        "Rows_Removed": removed_r1,
        "Rows_Remaining": len(current_df),
        "Pct_Dataset_Removed": (removed_r1 / initial_count) * 100
    })
    if verbose:
        print(f"• Rule 1 (Remove Duplicates):")
        print(f"    Removed : {removed_r1:7,d} rows")
        print(f"    Remaining: {len(current_df):7,d} rows")

    # Rule 2: Remove Cancellations (InvoiceNo starting with 'C')
    count_before = len(current_df)
    is_cancel = current_df['InvoiceNo'].astype(str).str.startswith('C')
    current_df = current_df[~is_cancel]
    removed_r2 = count_before - len(current_df)
    removal_logs.append({
        "Rule": "Rule 2: Remove Cancellations (InvoiceNo starting with 'C')",
        "Rows_Removed": removed_r2,
        "Rows_Remaining": len(current_df),
        "Pct_Dataset_Removed": (removed_r2 / initial_count) * 100
    })
    if verbose:
        print(f"\n• Rule 2 (Remove Cancellations starting with 'C'):")
        print(f"    Removed : {removed_r2:7,d} rows")
        print(f"    Remaining: {len(current_df):7,d} rows")

    # Rule 3: Remove Non-positive Quantity (Quantity <= 0)
    count_before = len(current_df)
    is_non_pos_qty = current_df['Quantity'] <= 0
    current_df = current_df[~is_non_pos_qty]
    removed_r3 = count_before - len(current_df)
    removal_logs.append({
        "Rule": "Rule 3: Remove Non-positive Quantity (Quantity <= 0)",
        "Rows_Removed": removed_r3,
        "Rows_Remaining": len(current_df),
        "Pct_Dataset_Removed": (removed_r3 / initial_count) * 100
    })
    if verbose:
        print(f"\n• Rule 3 (Remove Non-positive Quantity <= 0):")
        print(f"    Removed : {removed_r3:7,d} rows")
        print(f"    Remaining: {len(current_df):7,d} rows")

    # Rule 4: Remove Non-positive UnitPrice (UnitPrice <= 0)
    count_before = len(current_df)
    is_non_pos_price = current_df['UnitPrice'] <= 0
    current_df = current_df[~is_non_pos_price]
    removed_r4 = count_before - len(current_df)
    removal_logs.append({
        "Rule": "Rule 4: Remove Non-positive UnitPrice (UnitPrice <= 0)",
        "Rows_Removed": removed_r4,
        "Rows_Remaining": len(current_df),
        "Pct_Dataset_Removed": (removed_r4 / initial_count) * 100
    })
    if verbose:
        print(f"\n• Rule 4 (Remove Non-positive UnitPrice <= 0):")
        print(f"    Removed : {removed_r4:7,d} rows")
        print(f"    Remaining: {len(current_df):7,d} rows")

    # Rule 5: Remove Non-product StockCodes
    count_before = len(current_df)
    is_non_prod = is_non_product_code(current_df['StockCode'])
    current_df = current_df[~is_non_prod]
    removed_r5 = count_before - len(current_df)
    removal_logs.append({
        "Rule": "Rule 5: Remove Non-product StockCodes (POST, DOT, M, etc.)",
        "Rows_Removed": removed_r5,
        "Rows_Remaining": len(current_df),
        "Pct_Dataset_Removed": (removed_r5 / initial_count) * 100
    })
    if verbose:
        print(f"\n• Rule 5 (Remove Non-product StockCodes):")
        print(f"    Removed : {removed_r5:7,d} rows")
        print(f"    Remaining: {len(current_df):7,d} rows")

    # Final formatting & cleaning types
    current_df['StockCode'] = current_df['StockCode'].astype(str).str.strip()
    current_df['InvoiceNo'] = current_df['InvoiceNo'].astype(str).str.strip()
    current_df['Description'] = current_df['Description'].fillna("").astype(str).str.strip()
    current_df['InvoiceDate'] = pd.to_datetime(current_df['InvoiceDate'])
    current_df['Week_Start'] = current_df['InvoiceDate'].dt.to_period('W-SUN').dt.start_time

    removal_log_df = pd.DataFrame(removal_logs)
    total_removed = initial_count - len(current_df)

    if verbose:
        print("=" * 75)
        print("3. CLEANING SUMMARY")
        print("=" * 75)
        print(f"Initial raw rows  : {initial_count:7,d}")
        print(f"Total rows removed: {total_removed:7,d} ({(total_removed / initial_count)*100:.2f}%)")
        print(f"Final clean rows  : {len(current_df):7,d} ({(len(current_df) / initial_count)*100:.2f}%)")
        print(f"Unique products   : {current_df['StockCode'].nunique():7,d}")
        print(f"Date range        : {current_df['InvoiceDate'].min()} to {current_df['InvoiceDate'].max()}")
        print("\n• Post-cleaning null counts:")
        for col in current_df.columns:
            n_null = current_df[col].isnull().sum()
            print(f"    - {col:15s}: {n_null:7,d}")
        print("=" * 75)

    return current_df, removal_log_df


def _find_file(candidates: list) -> str:
    """Helper to locate a file from a list of plausible paths."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_raw_data(
    filepath: str = None,
    cache_path: str = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> pd.DataFrame:
    """
    Loads and cleans the UCI Online Retail dataset.
    
    Default paths:
    - Raw: 'data/raw/Online Retail.xlsx'
    - Processed cache: 'data/processed/data_clean.parquet'
    
    If cached parquet exists and use_cache=True, loads in ~0.1s.
    Otherwise, reads from raw Excel, applies clean_online_retail_data,
    and caches the result.
    """
    if filepath is None:
        filepath = _find_file([
            "data/raw/Online Retail.xlsx",
            "../data/raw/Online Retail.xlsx",
            "Online Retail.xlsx",
            "../Online Retail.xlsx"
        ])

    if cache_path is None:
        cache_path = _find_file([
            "data/processed/data_clean.parquet",
            "../data/processed/data_clean.parquet",
            "data_clean.parquet",
            "../data_clean.parquet"
        ])

    if use_cache and not force_reload and os.path.exists(cache_path):
        print(f"[cleaning.py] Loading cached cleaned data from '{cache_path}'...")
        df = pd.read_parquet(cache_path)
        print(f"[cleaning.py] Loaded {len(df):,} cleaned records from cache.")
        return df

    print(f"[cleaning.py] Reading raw Excel data from '{filepath}' (this takes ~25s)...")
    df_raw = pd.read_excel(filepath)
    print(f"[cleaning.py] Raw records loaded: {len(df_raw):,}")

    clean_df, _ = clean_online_retail_data(df_raw, verbose=False)

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        clean_df.to_parquet(cache_path, index=False)
        print(f"[cleaning.py] Successfully cached {len(clean_df):,} records to '{cache_path}'.")
    except Exception as e:
        print(f"[cleaning.py] Warning: Could not cache to parquet: {e}")

    return clean_df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generates a concise summary dictionary of dataset statistics.
    """
    return {
        "total_records": len(df),
        "unique_products": df['StockCode'].nunique(),
        "unique_invoices": df['InvoiceNo'].nunique(),
        "unique_customers": df['CustomerID'].dropna().nunique() if 'CustomerID' in df.columns else 0,
        "date_min": df['InvoiceDate'].min().strftime('%Y-%m-%d %H:%M'),
        "date_max": df['InvoiceDate'].max().strftime('%Y-%m-%d %H:%M'),
        "total_units_sold": int(df['Quantity'].sum()),
        "total_gross_revenue": float((df['Quantity'] * df['UnitPrice']).sum())
    }


def main():
    candidate_paths = [
        "data/raw/Online Retail.xlsx",
        "../data/raw/Online Retail.xlsx",
        "Online Retail.xlsx",
        "../Online Retail.xlsx"
    ]
    filepath = None
    for p in candidate_paths:
        if os.path.exists(p):
            filepath = p
            break

    if filepath is None:
        raise FileNotFoundError(f"Raw dataset not found in any expected location: {candidate_paths}")

    print(f"Loading '{filepath}' into memory (this takes ~25s)...")
    t0 = time.time()
    df_raw = pd.read_excel(filepath)
    print(f"Loaded in {time.time() - t0:.2f}s.\n")

    # Step 1: Audit
    audit_stats = audit_raw_data(df_raw)
    print_audit_report(audit_stats)

    # Step 2: Clean and record removals
    clean_df, removal_df = clean_online_retail_data(df_raw, verbose=True)

    # Save cleaned data to data/processed/ and removal log to artifacts/
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    removal_df.to_csv("artifacts/cleaning_removal_counts.csv", index=False)
    clean_df.to_parquet("data/processed/data_clean.parquet", index=False)
    print(f"\n[cleaning.py] Saved cleaned dataset to 'data/processed/data_clean.parquet'")
    print(f"[cleaning.py] Saved removal log to 'artifacts/cleaning_removal_counts.csv'")


if __name__ == "__main__":
    main()
