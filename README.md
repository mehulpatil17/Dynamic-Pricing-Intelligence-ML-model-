# ES 335 ML Project: Weekly Demand Forecasting & Pricing What-If Tool

An end-to-end Machine Learning pipeline and interactive Streamlit web application forecasting weekly product demand from the **UCI Online Retail dataset** and providing a scenario-based What-If Revenue Tool.

---

## 📌 1. Project Overview & Specifications

- **Course:** ES 335 Machine Learning Project
- **Dataset:** UCI Online Retail (`data/raw/Online Retail.xlsx`)
  - **Fields:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`
  - **Cost Column:** None present in the data (revenue is modeled directly, not profit).
- **Forecasting Objective:** Forecast weekly product demand ($Q_{i, t}$) for the top 100 merchandise products.
- **Baselines:**
  1. **Training Average per Product:** Historical mean weekly demand $\bar{y}_{i, \text{train}}$ observed in the training split.
  2. **Last-Week Demand:** Naive persistence forecast $y_{i, t-1}$ (`lag_1`).
- **Machine Learning Models:**
  1. **Linear Regression** (`sklearn.linear_model.LinearRegression`)
  2. **Random Forest Regressor** (`sklearn.ensemble.RandomForestRegressor`), tuned strictly on validation weeks.
- **Evaluation Metrics:**
  - **MAE (Mean Absolute Error):** $\frac{1}{N} \sum_{k=1}^N |y_k - \hat{y}_k|$
  - **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N} \sum_{k=1}^N (y_k - \hat{y}_k)^2}$
- **Split Scheme:** **Strict Chronological 70/15/15 Split by Week** (zero temporal leakage, no random shuffling).
- **Pricing Tool:** **What-If Revenue Simulation** ($\text{Revenue} = \text{Price} \times \hat{Q}_{\text{predicted}}$), explicitly **not causal**.
- **Dashboard:** Streamlit application (`app/streamlit_app.py`) that exclusively loads precomputed predictions and metrics.

---

## 🏗️ 2. Repository Architecture

```
project/
├── data/
│   ├── raw/                 # Online Retail.xlsx (unchanged)
│   └── processed/           # cleaned transactions, product-week table
├── notebooks/
│   ├── 01_inspection.ipynb
│   ├── 02_eda.ipynb
│   └── 03_price_variation.ipynb
├── src/
│   ├── cleaning.py
│   ├── features.py
│   ├── models.py            # baselines, LinearRegression, RandomForest
│   ├── evaluate.py          # MAE, RMSE, chronological split
│   ├── pricing.py           # variation check, scenarios
│   └── run_pipeline.py      # writes predictions and metrics to disk
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_features.py
│   └── test_split.py
├── reports/
│   ├── figures/
│   └── final_report.md
├── requirements.txt
└── README.md
```

---

## 🚀 3. Step-by-Step Reproduction Guide (Raw File to Dashboard)

Follow these exact steps from your terminal to reproduce the entire pipeline from `data/raw/Online Retail.xlsx` to the web dashboard.

### Prerequisites & Environment Setup

```powershell
# 1. Navigate to project root
cd path/to/project

# 2. Verify Python environment
py --version

# 3. Install required dependencies
pip install -r requirements.txt
```

---

### Step 1: Clean Raw Transactions (`src/cleaning.py`)
Loads `data/raw/Online Retail.xlsx`, prints the initial audit, and applies 5 sequential cleaning rules:

```powershell
py src/cleaning.py
```

**Verified Row Removal Counts:**
- Initial raw records: **541,909**
- Rule 1 (Exact Duplicates): Removed **5,268 rows** $\rightarrow$ 536,641 remaining
- Rule 2 (Cancellations with 'C'): Removed **9,251 rows** $\rightarrow$ 527,390 remaining
- Rule 3 (Quantity $\le 0$): Removed **1,336 rows** $\rightarrow$ 526,054 remaining
- Rule 4 (UnitPrice $\le 0$): Removed **1,176 rows** $\rightarrow$ 524,878 remaining
- Rule 5 (Non-Product StockCodes like 'POST', 'M', 'D'): Removed **2,341 rows** $\rightarrow$ 522,537 remaining
- **Final Clean Rows:** **522,537** (96.43% retained) across **3,907 unique products**.
- Output saved to `data/processed/data_clean.parquet` (~0.1s fast loading).

---

### Step 2: Price Variation Check & EDA (`src/pricing.py` & `notebooks/`)
Evaluates active weeks with sales, distinct weekly prices, weeks per price level, and min/max price ratios using **training weeks only**:

```powershell
py src/pricing.py
```
- Applies stability thresholds: $\ge 20$ active sales weeks, $\ge 500$ training units, min/max price ratio $\ge 0.50$, and $\ge 1.0$ weeks per price level.
- Filters **515 products** passing stability criteria.
- Jupyter notebooks available in `notebooks/`:
  - `notebooks/01_inspection.ipynb`: Data inspection and cleaning rules audit.
  - `notebooks/02_eda.ipynb`: Weekly total units, top 30 products, price distributions.
  - `notebooks/03_price_variation.ipynb`: Training-only price stability audit and passed products.

---

### Step 3: Feature Engineering & Chronological Split (`src/features.py`)
Constructs the continuous product-week table, zero-fills missing weeks, and engineers strictly causal features:

```powershell
py src/features.py
```

**Engineering Specifications:**
- Top 100 volume products across 54 calendar weeks = **5,400 balanced product-week records**.
- Zero-demand weeks filled: **1,857 / 5,400 (34.4%)**.
- Features engineered: `lag_1`, `lag_2`, `lag_4`, `rolling_mean_4` (shifted by 1 so current week $t$ is strictly excluded), `rolling_std_4`, `week_of_year`, `mean_unit_price`.
- Warm-up period (first 4 weeks) dropped $\rightarrow$ **50 modeling weeks (5,000 observations)**.
- **Strict Chronological 70/15/15 Split:**
  - **Train (70%):** 35 weeks (`2010-12-27` to `2011-08-22`), 3,500 rows.
  - **Validation (15%):** 8 weeks (`2011-08-29` to `2011-10-17`), 800 rows — used strictly for tuning Random Forest.
  - **Test (15%):** 7 weeks (`2011-10-24` to `2011-12-05`), 700 rows — unseen holdout.
  - Boundary: $\max(\text{Train}) < \min(\text{Val}) \le \max(\text{Val}) < \min(\text{Test})$.

---

### Step 4: Model Training & Validation Tuning (`src/models.py`)
Fits Baselines, Ordinary Least Squares Linear Regression, and tunes Random Forest strictly on validation weeks:

```powershell
py src/models.py
```

**Random Forest Validation Tuning Grid (Evaluated strictly on the 8 Validation Weeks):**
- Candidate 1: `n_est=50, max_depth=6, min_leaf=2` $\rightarrow$ Val MAE: **177.91**, Val RMSE: **288.08** (**Selected Best**)
- Candidate 2: `n_est=50, max_depth=10, min_leaf=2` $\rightarrow$ Val MAE: 181.64, Val RMSE: 290.22
- Candidate 3: `n_est=100, max_depth=6, min_leaf=2` $\rightarrow$ Val MAE: 178.23, Val RMSE: 288.56
- Candidate 4: `n_est=100, max_depth=10, min_leaf=1` $\rightarrow$ Val MAE: 183.92, Val RMSE: 293.40
- Candidate 5: `n_est=100, max_depth=10, min_leaf=2` $\rightarrow$ Val MAE: 182.34, Val RMSE: 290.99
- Candidate 6: `n_est=100, max_depth=None, min_leaf=2` $\rightarrow$ Val MAE: 190.43, Val RMSE: 294.41

Exports single unified predictions table: `artifacts/unified_test_predictions.csv` (700 rows).

---

### Step 5: Held-Out Test Evaluation — Evaluated Only Once (`src/evaluate.py`)
Computes out-of-sample MAE and RMSE on the held-out 7 test weeks:

```powershell
py src/evaluate.py
```

**Verified Test Set Results (All 700 Observations):**

| Model / Baseline | Test MAE | Test RMSE | Description |
| :--- | :---: | :---: | :--- |
| **Model: Linear Regression** | **355.52** | **3101.94** | **Best overall out-of-sample performance** |
| Baseline: Last-Week Demand | 390.95 | 3109.43 | Naive lag-1 persistence |
| Model: Random Forest (Tuned) | 396.51 | 3116.72 | Regularized ensemble ($D=6, L=2, T=50$) |
| Baseline: Training Average | 446.89 | 3132.26 | Training mean per product ($\bar{y}_{i, \text{train}}$) |

**Diagnostic Sensitivity Note:** In Week 7 (`2011-12-05`), StockCode `23843` (*PAPER CRAFT , LITTLE BIRDIE*) experienced a single anomalous bulk order of 80,995 units. Excluding this single wholesale order (693 rows):
- **Linear Regression:** **MAE = 242.10 | RMSE = 503.13** (Best)
- **Last-Week Demand:** **MAE = 278.02 | RMSE = 547.58**
- **Random Forest:** **MAE = 283.47 | RMSE = 592.78**
- **Training Average:** **MAE = 334.52 | RMSE = 666.18**

Generates figures in `reports/figures/`:
1. `actual_vs_predicted_weekly_aggregate.png`
2. `error_by_week.png`
3. `actual_vs_predicted_top_products.png`
4. `per_product_error_distribution.png`

---

### Step 6: Pricing What-If Simulation (`src/pricing.py`)
Generates candidate prices inside $[P_5, P_{95}]$, holds non-price features fixed, predicts demand, and calculates $\text{Revenue} = \text{Price} \times \hat{Q}$:

```powershell
py src/pricing.py
```

**Monotonicity Findings across 62 Passed Products in Panel:**
- **Linear Regression:** **0 / 62 warnings**. Because global $\beta_{\text{price}} = -19.67 < 0$, demand decreases monotonically with price across all passed products.
- **Random Forest:** **13 / 62 products trigger rising demand warnings**. Observational step splits associate certain higher price tiers with high-volume peak seasonal weeks, demonstrating the non-causal nature of observational regression. Flagged items include `22197`, `16014`, `22469`, `22178`, `21212`, `22492`, `21931`, `21790`, `23201`, `20713`, `84945`, `82482`, `22457`.

---

### Step 7: Launch the Interactive Dashboard (`app/streamlit_app.py`)
Launches the Streamlit application that exclusively loads the precomputed artifacts:

```powershell
py -m streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`. Features include:
- Persistent methodological warning banner.
- Product selector with searchable `StockCode` and `Description`.
- 54-week complete historical demand timeline.
- Out-of-sample actual vs. predicted comparison line charts.
- Overall benchmark and per-product error tables.
- Candidate pricing grid ($[P_5, P_{95}]$) with dynamic monotonicity alerts.

---

### ⚡ One-Command Full Pipeline Runner

Alternatively, run the complete end-to-end pipeline from data cleaning through model evaluation and pricing simulation in a single command:

```powershell
py src/run_pipeline.py
```
Execution completes in approximately **6 seconds**.

---

## 🧪 4. Test Suite Verification

A comprehensive pytest test suite of **34 unit tests** covers every module in the project:

```powershell
# Run the complete test suite
py -m pytest tests/ -v
```

### Test Coverage Summary:
- [`tests/test_features.py`](file:///c:/Users/MEHUL/OneDrive/Documents/study/ML/project/tests/test_features.py) (9 tests):
  - Continuous panel zero-filling
  - Zero current-week lookahead leakage
  - Zero future-week lookahead leakage
  - Exact mathematical lag and rolling values
  - Strict chronological split boundaries
  - Non-merchandise code detection
  - Raw audit statistics calculation
  - Sequential cleaning rule row removals
  - Cleaned dataset invariants (no negatives, no cancellations, no service codes)
- [`tests/test_split.py`](file:///c:/Users/MEHUL/OneDrive/Documents/study/ML/project/tests/test_split.py) (25 tests):
  - Chronological split temporal ordering ($\max(\text{train}) < \min(\text{val}) \le \max(\text{val}) < \min(\text{test})$)
  - Zero week overlap across partitions
  - 70/15/15 proportions calculation
  - Row count conservation
  - Insufficient weeks error handling
  - Baseline 1 training average mathematical exactness
  - Baseline 2 last-week persistence
  - Linear Regression fitting and non-negative clipping
  - Random Forest tuning on validation weeks only
  - Unified predictions table schema
  - Overall MAE/RMSE calculations
  - Per-product metrics and best model selection
  - Error-by-week aggregation
  - Model ranking summaries
  - Evaluation plots file generation
  - Evaluation artifacts CSV export
  - 5th to 95th percentile price bounds
  - Candidate price grid edge cases
  - Revenue formula calculation ($\text{Price} \times \text{Demand}$)
  - Downward slope acceptance
  - Upward slope monotonicity warning trigger
  - Holding non-price features fixed during prediction
  - End-to-end single-product simulation
  - Dashboard path resolution
  - Instant dashboard artifact loading

**Result:** `34 passed in ~13s` with zero warnings and zero failures.

---

## ⚠️ 5. Methodological Notice: Pricing Tool Semantics

> [!WARNING]
> **Pricing is What-If Simulation, NOT Causal Elasticity**:
>
> In observational retail transaction data, prices are set by retail managers based on customer demand, supplier cost shifts, inventory clear-outs, and holiday promotional calendars. Because this dataset lacks exogenous randomized price variation, our models estimate **predictive conditional demand** ($\hat{Q} \mid P, X_{\text{lags}}$), **not causal structural elasticity** ($\frac{\partial Q}{\partial P}$).
>
> When the dashboard simulates revenue:
> $$\text{Simulated Revenue} = P_{\text{candidate}} \times \hat{Q}(P_{\text{candidate}})$$
> it is exploring the model's conditional prediction holding other historical lags fixed. The warning banner displayed in both terminal and web UI alerts the user if non-causal regression artifacts produce upward demand slopes.

---

## 📋 6. Summary Checklist of What to Verify

| Item to Verify | Expected Value / File Location | Verification Command |
| :--- | :--- | :--- |
| **Data Cleaning** | 522,537 rows retained (96.4%) | `py src/cleaning.py` |
| **Chronological Split** | Train (35 wks) < Val (8 wks) < Test (7 wks) | `py src/features.py` |
| **RF Validation Tuning** | `n_est=50, max_depth=6, min_leaf=2` (Val MAE: 177.91) | `py src/models.py` |
| **Single Test Evaluation** | Linear Regression Test MAE = **355.52** | `py src/evaluate.py` |
| **Unified Table** | 700 rows, 4 predictions per row | [`artifacts/unified_test_predictions.csv`](file:///c:/Users/MEHUL/OneDrive/Documents/study/ML/project/artifacts/unified_test_predictions.csv) |
| **Pricing Monotonicity** | LR: 0 warnings; RF: 13 warnings | `py src/pricing.py` |
| **Dashboard** | Instant load (<0.5s), persistent warning banner | `py -m streamlit run app/streamlit_app.py` |
| **Unit Test Suite** | 34 tests passing, 0 failures | `py -m pytest tests/ -v` |
