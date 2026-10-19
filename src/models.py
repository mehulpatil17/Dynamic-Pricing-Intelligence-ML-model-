"""
src/models.py
Module implementing forecasting models, baselines, validation-based tuning,
and unified prediction tables.

Baselines:
1. Training Average per Product (mean weekly demand in training set)
2. Last-Week Demand (naive lag-1 forecast)

Machine Learning Models:
1. Linear Regression (OLS)
2. Random Forest Regressor (tuned strictly on validation weeks)

Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


# Default compact parameter grid for tuning Random Forest on validation weeks
DEFAULT_RF_PARAM_GRID: List[Dict[str, Any]] = [
    {'n_estimators': 50,  'max_depth': 6,    'min_samples_leaf': 2},
    {'n_estimators': 50,  'max_depth': 10,   'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 6,    'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': 10,   'min_samples_leaf': 1},
    {'n_estimators': 100, 'max_depth': 10,   'min_samples_leaf': 2},
    {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 2},
]


def predict_baseline_train_avg(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    product_col: str = 'StockCode',
    target_col: str = 'weekly_quantity'
) -> np.ndarray:
    """
    Baseline 1: Predicts the training-set mean weekly demand for each product.
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training set containing historical demand.
    eval_df : pd.DataFrame
        Evaluation set (validation or test).
    product_col : str
        Product identifier column ('StockCode').
    target_col : str
        Target column ('weekly_quantity').
        
    Returns
    -------
    np.ndarray
        Non-negative demand predictions vector.
    """
    train_means = train_df.groupby(product_col)[target_col].mean()
    overall_mean = float(train_df[target_col].mean())
    preds = eval_df[product_col].map(train_means).fillna(overall_mean).values
    return np.clip(preds, 0.0, None)


def predict_baseline_last_week(
    eval_df: pd.DataFrame,
    lag_col: str = 'lag_1'
) -> np.ndarray:
    """
    Baseline 2: Predicts the immediately preceding week's observed demand (lag-1 naive persistence).
    
    Parameters
    ----------
    eval_df : pd.DataFrame
        Evaluation set containing 'lag_1'.
    lag_col : str
        Lag-1 column name.
        
    Returns
    -------
    np.ndarray
        Non-negative demand predictions vector.
    """
    return np.clip(eval_df[lag_col].values, 0.0, None)


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LinearRegression:
    """
    Fits an Ordinary Least Squares Linear Regression model on training features.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target demand.
        
    Returns
    -------
    LinearRegression
        Fitted scikit-learn LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def tune_random_forest_on_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    param_grid: Optional[List[Dict[str, Any]]] = None,
    metric: str = 'mae',
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[RandomForestRegressor, Dict[str, Any], pd.DataFrame]:
    """
    Tunes Random Forest hyperparameters strictly on validation weeks using a grid search.
    
    Methodology:
    1. Fits candidate parameters on (X_train, y_train).
    2. Evaluates out-of-sample predictions on (X_val, y_val).
    3. Selects the hyperparameter set with the lowest validation MAE.
    
    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training features and target.
    X_val, y_val : pd.DataFrame, pd.Series
        Validation features and target (strictly chronological validation weeks).
    param_grid : Optional[List[Dict[str, Any]]]
        List of parameter dictionaries to evaluate.
    metric : str
        Metric used for model selection ('mae' or 'rmse').
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Whether to print validation evaluation scores for each candidate.
        
    Returns
    -------
    Tuple[RandomForestRegressor, Dict[str, Any], pd.DataFrame]
        best_model, best_params, tuning_history_df
    """
    if param_grid is None:
        param_grid = DEFAULT_RF_PARAM_GRID

    tuning_history: List[Dict[str, Any]] = []
    best_score = float('inf')
    best_params: Optional[Dict[str, Any]] = None
    best_model: Optional[RandomForestRegressor] = None

    if verbose:
        print("\n" + "=" * 70)
        print("TUNING RANDOM FOREST (ON VALIDATION WEEKS ONLY)")
        print("=" * 70)

    for idx, params in enumerate(param_grid, start=1):
        rf_candidate = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
        rf_candidate.fit(X_train, y_train)

        # Predict on validation weeks
        val_preds = np.clip(rf_candidate.predict(X_val), 0.0, None)
        val_mae = float(mean_absolute_error(y_val, val_preds))
        val_rmse = float(root_mean_squared_error(y_val, val_preds))

        score_to_compare = val_mae if metric == 'mae' else val_rmse
        is_best = score_to_compare < best_score
        if is_best:
            best_score = score_to_compare
            best_params = params
            best_model = rf_candidate

        record = {
            'Candidate': idx,
            'n_estimators': params['n_estimators'],
            'max_depth': str(params['max_depth']),
            'min_samples_leaf': params['min_samples_leaf'],
            'Val_MAE': round(val_mae, 4),
            'Val_RMSE': round(val_rmse, 4),
            'Is_Best': is_best
        }
        tuning_history.append(record)

        if verbose:
            print(f"[{idx}/{len(param_grid)}] Params: {params} | Val MAE: {val_mae:7.2f} | Val RMSE: {val_rmse:7.2f} {'<-- Best so far' if is_best else ''}")

    tuning_df = pd.DataFrame(tuning_history)
    if verbose:
        print("=" * 70)
        print(f"Best RF Parameters: {best_params} (Validation {metric.upper()}: {best_score:.2f})")
        print("=" * 70 + "\n")

    return best_model, best_params, tuning_df


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Computes MAE and RMSE metrics for a given model prediction vector.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4)
    }


def build_unified_predictions_table(
    eval_df: pd.DataFrame,
    pred_train_avg: np.ndarray,
    pred_last_week: np.ndarray,
    pred_lr: np.ndarray,
    pred_rf: np.ndarray,
    product_col: str = 'StockCode',
    date_col: str = 'week_start',
    target_col: str = 'weekly_quantity',
    desc_col: str = 'description'
) -> pd.DataFrame:
    """
    Combines ground truth demand and predictions from all 4 methods into a single unified table.
    
    Columns returned:
    - StockCode: Product identifier
    - week_start: Monday week start date
    - description: Product name
    - weekly_quantity: Actual observed demand
    - pred_train_avg: Baseline 1 (Training Average)
    - pred_last_week: Baseline 2 (Last-Week Demand)
    - pred_linear_regression: Model 1 (Linear Regression)
    - pred_random_forest: Model 2 (Tuned Random Forest)
    """
    table = eval_df[[product_col, date_col]].copy()
    if desc_col in eval_df.columns:
        table[desc_col] = eval_df[desc_col]
    elif 'Description' in eval_df.columns:
        table[desc_col] = eval_df['Description']
    else:
        table[desc_col] = ""

    table[target_col] = eval_df[target_col].values
    table['pred_train_avg'] = np.clip(pred_train_avg, 0.0, None)
    table['pred_last_week'] = np.clip(pred_last_week, 0.0, None)
    table['pred_linear_regression'] = np.clip(pred_lr, 0.0, None)
    table['pred_random_forest'] = np.clip(pred_rf, 0.0, None)

    return table.sort_values([product_col, date_col]).reset_index(drop=True)


def train_and_evaluate_all_methods(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'weekly_quantity',
    rf_param_grid: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    End-to-end training and evaluation:
    1. Baseline 1: Training Average
    2. Baseline 2: Last-Week Demand
    3. Model 1: Linear Regression
    4. Model 2: Random Forest (tuned on validation weeks)
    5. Returns single unified predictions table for Test (and Validation).
    
    Returns
    -------
    Dict[str, Any] containing:
    - 'val_predictions': pd.DataFrame (unified predictions table on validation set)
    - 'test_predictions': pd.DataFrame (unified predictions table on test set)
    - 'val_metrics': pd.DataFrame (MAE & RMSE on validation set)
    - 'test_metrics': pd.DataFrame (MAE & RMSE on test set)
    - 'best_rf_params': Dict[str, Any]
    - 'rf_tuning_history': pd.DataFrame
    - 'models': Dict of fitted estimators ('linear_regression', 'random_forest')
    """
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # 1. Baseline Predictions
    val_p_train_avg = predict_baseline_train_avg(train_df, val_df, target_col=target_col)
    test_p_train_avg = predict_baseline_train_avg(train_df, test_df, target_col=target_col)

    val_p_last_week = predict_baseline_last_week(val_df)
    test_p_last_week = predict_baseline_last_week(test_df)

    # 2. Linear Regression
    lr_model = train_linear_regression(X_train, y_train)
    val_p_lr = np.clip(lr_model.predict(X_val), 0.0, None)
    test_p_lr = np.clip(lr_model.predict(X_test), 0.0, None)

    # 3. Random Forest - Tune strictly on validation weeks
    best_rf, best_rf_params, tuning_df = tune_random_forest_on_validation(
        X_train, y_train, X_val, y_val,
        param_grid=rf_param_grid,
        metric='mae',
        verbose=verbose
    )
    val_p_rf = np.clip(best_rf.predict(X_val), 0.0, None)
    test_p_rf = np.clip(best_rf.predict(X_test), 0.0, None)

    # 4. Build Unified Predictions Tables
    val_preds_table = build_unified_predictions_table(
        val_df, val_p_train_avg, val_p_last_week, val_p_lr, val_p_rf, target_col=target_col
    )
    test_preds_table = build_unified_predictions_table(
        test_df, test_p_train_avg, test_p_last_week, test_p_lr, test_p_rf, target_col=target_col
    )

    # 5. Compute Metrics
    val_metrics = pd.DataFrame([
        evaluate_predictions(y_val.values, val_p_train_avg, "Baseline: Training Average"),
        evaluate_predictions(y_val.values, val_p_last_week, "Baseline: Last-Week Demand"),
        evaluate_predictions(y_val.values, val_p_lr, "Model: Linear Regression"),
        evaluate_predictions(y_val.values, val_p_rf, "Model: Random Forest (Tuned)")
    ])

    test_metrics = pd.DataFrame([
        evaluate_predictions(y_test.values, test_p_train_avg, "Baseline: Training Average"),
        evaluate_predictions(y_test.values, test_p_last_week, "Baseline: Last-Week Demand"),
        evaluate_predictions(y_test.values, test_p_lr, "Model: Linear Regression"),
        evaluate_predictions(y_test.values, test_p_rf, "Model: Random Forest (Tuned)")
    ])

    return {
        'val_predictions': val_preds_table,
        'test_predictions': test_preds_table,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'best_rf_params': best_rf_params,
        'rf_tuning_history': tuning_df,
        'models': {
            'linear_regression': lr_model,
            'random_forest': best_rf
        }
    }
