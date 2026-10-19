# ES 335 ML Project: Weekly Demand Forecasting & What-If Revenue Tool
**Final Technical Report**  
*Department of Computer Science & Engineering / Electrical Engineering*  
*Course: ES 335 Machine Learning*  

---

## Executive Summary

This report documents the design, implementation, and empirical evaluation of an end-to-end Machine Learning pipeline and interactive Streamlit web application forecasting weekly product demand from the **UCI Online Retail dataset** (`Online Retail.xlsx`). In addition, the project develops an empirical **What-If Revenue Tool** evaluated within the 5th and 95th percentiles of observed prices.

### Key Empirical Findings:
1. **Cleaned Foundation:** Starting from 541,909 raw records, sequential cleaning rules removed 19,372 invalid rows (duplicates, cancellations, negative quantities, zero prices, non-product service codes), retaining **522,537 valid transactions (96.4%)** across **3,907 unique products**.
2. **Causal Time-Series Panel:** A continuous, balanced product-week panel of the top 100 volume products across 54 calendar weeks was constructed, with missing sales weeks zero-filled (1,857 / 5,400 weeks, 34.4%). Causal lag and rolling features were engineered with zero future lookahead leakage, verified mathematically by pytest.
3. **Strict Chronological Holdout:** The timeline was partitioned strictly into **70% Train** (35 weeks, 3,500 rows), **15% Validation** (8 weeks, 800 rows), and **15% Test** (7 weeks, 700 rows).
4. **Validation-Based Hyperparameter Tuning:** Random Forest hyperparameters were tuned strictly on the 8 validation weeks (`n_estimators=50, max_depth=6, min_samples_leaf=2`, Val MAE: **177.91**), avoiding overfitting on volatile retail demand.
5. **Single-Run Test Benchmark:** Evaluated **strictly once** on the held-out test split, **Ordinary Least Squares (OLS) Linear Regression achieved the best overall performance (Test MAE: 355.52 | Test RMSE: 3101.94)**, outperforming Random Forest (MAE: 396.51), Last-Week Demand (MAE: 390.95), and Training Average (MAE: 446.89).
6. **Diagnostic Outlier Impact:** In the final test week (`2011-12-05`), StockCode `23843` (*PAPER CRAFT , LITTLE BIRDIE*) experienced a known single wholesale bulk transaction of **80,995 units**. Excluding this single extreme outlier, Linear Regression achieved **MAE = 242.10 | RMSE = 503.13**.
7. **Pricing Tool Semantics & Monotonicity Warnings:** Candidate prices were evaluated inside the $[P_5, P_{95}]$ range holding other features fixed. The tool incorporates an explicit warning when predicted demand rises with price: Linear Regression produced **0 / 62 warnings** ($\beta_{\text{price}} < 0$), whereas Random Forest triggered **13 / 62 warnings** due to observational decision splits where peak seasonal shopping coincided with higher prices.
8. **Interactive Web Dashboard:** A Streamlit dashboard (`app/streamlit_app.py`) was deployed that exclusively loads precomputed artifacts with an always-visible warning banner.

---

## 1. Dataset & Data Cleaning Audit (`src/cleaning.py`)

### 1.1 Dataset Schema & Constraints
The UCI Online Retail dataset contains transactional sales for a UK-based registered online retailer between December 1, 2010 and December 9, 2011.
- **Fields Present:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
- **Cost Column:** **No cost column exists** in the dataset. Therefore, financial analysis models gross revenue directly, not net margin.

### 1.2 Initial Audit (Raw Data)
Before applying any cleaning rules, a baseline audit of all 541,909 raw records was executed:
- **Missing Values:** `Description` had 1,454 missing entries (0.27%); `CustomerID` had 135,080 missing entries (24.93%). Because forecasting is performed per product rather than per user, missing customer IDs were retained for valid merchandise sales.
- **Duplicate Rows:** 5,268 exact duplicate rows were detected (0.97%).
- **Cancellations:** 9,251 rows had invoice numbers starting with `'C'` (1.71%).
- **Non-Positive Values:** 10,624 rows had Quantity $\le 0$; 2,515 rows had UnitPrice $\le 0$.
- **Non-Product Codes:** 2,759 rows contained non-merchandise administrative codes (`POST`, `DOT`, `M`, `D`, `C2`, `CRUK`, `BANK CHARGES`, `PADS`, `gift_*`).

### 1.3 Sequential Cleaning Pipeline & Verified Removal Counts
To eliminate order dependency ambiguity, cleaning rules were applied sequentially:

```
Raw Dataset (Online Retail.xlsx): 541,909 rows
   │
   ├── [Rule 1] Drop Exact Duplicate Rows
   │            Removed: 5,268 rows  ──>  536,641 remaining
   │
   ├── [Rule 2] Remove Cancellations (InvoiceNo starting with 'C')
   │            Removed: 9,251 rows  ──>  527,390 remaining
   │
   ├── [Rule 3] Remove Non-Positive Quantity (Quantity <= 0)
   │            Removed: 1,336 rows  ──>  526,054 remaining
   │
   ├── [Rule 4] Remove Non-Positive UnitPrice (UnitPrice <= 0)
   │            Removed: 1,176 rows  ──>  524,878 remaining
   │
   └── [Rule 5] Remove Non-Product StockCodes (POST, DOT, M, D, C2, etc.)
                Removed: 2,341 rows  ──>  522,537 remaining
```

**Final Cleaned Dataset:** **522,537 rows** (96.43% retained) across **3,907 unique products**. Exported to `data/processed/data_clean.parquet` (~0.1s load time).

---

## 2. Exploratory Data Analysis & Price Variation Audit

### 2.1 Overall Sales Timeline & Seasonality
Aggregating sales by calendar week (Monday start) reveals strong retail seasonality:
- Early 2011 demand averaged **40,000 to 60,000 units/week**.
- Starting in October 2011, Q4 pre-holiday wholesale and retail purchasing expanded dramatically, peaking at **>140,000 units/week** in late November.
- *Reference Figure:* `reports/figures/weekly_total_units.png`

### 2.2 Product Volume Concentration
A small fraction of merchandise items drives the majority of unit volume:
- Top products include `85099B` (*JUMBO BAG RED RETROSPOT*, ~36k units), `22197` (*SMALL POPCORN HOLDER*, ~32k units), `84077` (*WORLD WAR 2 GLIDERS*, ~41k units), and `85123A` (*WHITE HANGING HEART T-LIGHT HOLDER*, ~30k units).
- *Reference Figure:* `reports/figures/top_30_products.png`

### 2.3 Price Distribution
- Unit prices are heavily right-skewed. The median unit selling price is **£2.08**, with over 90% of transactions priced under £5.00.
- *Reference Figure:* `reports/figures/price_distribution.png`

### 2.4 Training-Set Price Stability Check (`src/price_check.py`)
To ensure price variation is meaningful for downstream pricing scenarios, price stability metrics were computed **strictly on training weeks**:
1. `Weeks_With_Sales`: Active calendar weeks with $\ge 1$ transaction.
2. `Distinct_Weekly_Prices`: Count of distinct nominal price points.
3. `Weeks_Per_Price`: Average duration (in weeks) spent at each price level.
4. `Min_Max_Price_Ratio`: $\frac{\text{Min Price}}{\text{Max Price}}$ (measures relative volatility).

**Filtering Thresholds:**
- $\ge 20$ active sales weeks in training split
- $\ge 500$ cumulative training units sold
- Min/Max price ratio $\ge 0.50$ (price varied by at most 2x)
- $\ge 1.0$ weeks per price level

A total of **515 products** passed all stability criteria (`artifacts/price_check_passed_products.csv`). When matched against our top 100 modeled volume panel, **62 products** passed the audit, providing the empirical foundation for what-if candidate price ranges.

---

## 3. Time-Series Feature Engineering & Chronological Split (`src/features.py`)

### 3.1 Continuous Balanced Product-Week Panel
In transactional retail data, weeks with zero purchases are absent from raw tables. To avoid survivor bias, a continuous Cartesian grid was formed:

$$\mathcal{P} = \{\text{Top 100 Products}\} \times \{\text{54 Calendar Weeks}\} = 5,400\text{ observations}$$

- Missing weeks were explicitly filled with **`weekly_quantity = 0.0`**.
- Exactly **1,857 out of 5,400 product-weeks (34.4%)** had zero sales and were correctly imputed.
- Missing weekly unit prices during inactive weeks were imputed via forward-fill, then backward-fill, then overall historical product mean.

### 3.2 Causal Feature Engineering (Zero Lookahead Leakage)
To guarantee strict forecasting realism, all features predict demand at week $t$ using information available strictly before week $t$:
1. `lag_1`: Observed demand at week $t-1$.
2. `lag_2`: Observed demand at week $t-2$.
3. `lag_4`: Observed demand at week $t-4$.
4. `rolling_mean_4`: Trailing moving average over the past 4 weeks, **shifted by 1 week**:
   $$\text{rolling\_mean\_4}_t = \frac{1}{4} \sum_{k=1}^4 Q_{t-k}$$
5. `rolling_std_4`: Trailing standard deviation over the past 4 weeks (shifted by 1).
6. `week_of_year`: Calendar week integer (1 to 52) capturing annual seasonality.
7. `mean_unit_price`: Unit selling price.

Warm-up rows (first 4 weeks) where lag/rolling features are undefined were dropped, leaving **50 modeling weeks (5,000 balanced observations)**. Zero lookahead leakage was verified by 5 unit tests in `tests/test_features.py`.

### 3.3 Strict Chronological 70/15/15 Split
The 50 modeling weeks were partitioned strictly by calendar time:

```
[------------------ TRAIN (70%) ------------------][--- VAL (15%) ---][--- TEST (15%) ---]
 35 Weeks (2010-12-27 to 2011-08-22)                8 Weeks             7 Weeks
 3,500 Observations                                 800 Observations    700 Observations
```

- **Temporal Order Invariant:** $\max(\text{Train}) < \min(\text{Val}) \le \max(\text{Val}) < \min(\text{Test})$.
- **No Week Overlap:** $\text{Train} \cap \text{Val} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset$.
- Proved and enforced by unit tests in `tests/test_split.py`.

---

## 4. Model Training & Validation Tuning (`src/models.py`)

### 4.1 Forecasting Methods Formulated
1. **Baseline 1 (Training Average):** Predicts the constant mean weekly demand observed in the training split for each product:
   $$\hat{y}_{i, t}^{\text{TrainAvg}} = \bar{y}_{i, \text{train}}$$
2. **Baseline 2 (Last-Week Demand):** Naive persistence forecasting:
   $$\hat{y}_{i, t}^{\text{LastWeek}} = y_{i, t-1} = \text{lag\_1}$$
3. **Model 1 (Linear Regression):** Ordinary Least Squares (OLS) with non-negative prediction clipping $\max(0, \hat{y})$.
4. **Model 2 (Random Forest Regressor):** Non-linear ensemble with hyperparameter tuning evaluated strictly on the 8 validation weeks.

### 4.2 Hyperparameter Tuning on Validation Weeks
A compact grid was evaluated strictly against out-of-sample Validation MAE:

| Candidate | `n_estimators` | `max_depth` | `min_samples_leaf` | Validation MAE | Validation RMSE | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **50** | **6** | **2** | **177.91** | **288.08** | **Selected Best** |
| 2 | 50 | 10 | 2 | 181.64 | 290.22 | |
| 3 | 100 | 6 | 2 | 178.23 | 288.56 | |
| 4 | 100 | 10 | 1 | 183.92 | 293.40 | |
| 5 | 100 | 10 | 2 | 182.34 | 290.99 | |
| 6 | 100 | None | 2 | 190.43 | 294.41 | Overfitting |

*Methodological Insight:* Restricting tree depth (`max_depth=6`) regularized against extreme demand spikes, outperforming unconstrained trees (`max_depth=None`, Val MAE 190.43).

### 4.3 Validation Set Benchmark Performance (8 Weeks, 800 Observations)

| Model / Baseline | Val MAE (Units) | Val RMSE (Units) | Improvement vs Train Avg |
| :--- | :---: | :---: | :---: |
| **Model: Linear Regression** | **172.72** | **299.02** | **+18.2%** |
| **Model: Random Forest (Tuned)** | **177.91** | **288.08** | **+15.8%** |
| Baseline: Training Average | 211.28 | 381.25 | Reference (0.0%) |
| Baseline: Last-Week Demand | 224.81 | 374.49 | -6.4% |

---

## 5. Held-Out Test Set Evaluation (`src/evaluate.py`)

### 5.1 Protocol: Evaluated Only Once
To maintain test set integrity, model architectures and hyperparameters were fixed. The test set was evaluated **strictly once**, without retrospective tuning.

### 5.2 Overall Test Set Benchmark (All 700 Observations)

| Model / Baseline | Test MAE | Test RMSE | Description |
| :--- | :---: | :---: | :--- |
| **Model: Linear Regression** | **355.52** | **3101.94** | **Best overall out-of-sample forecast** |
| Baseline: Last-Week Demand | 390.95 | 3109.43 | Naive persistence |
| Model: Random Forest (Tuned) | 396.51 | 3116.72 | Regularized ensemble |
| Baseline: Training Average | 446.89 | 3132.26 | Historical training mean |

*Reference Figures:* `reports/figures/actual_vs_predicted_weekly_aggregate.png` and `reports/figures/per_product_error_distribution.png`

### 5.3 Diagnostic Outlier Analysis (StockCode 23843)
In the final week (`2011-12-05`), StockCode `23843` (*PAPER CRAFT , LITTLE BIRDIE*) logged a known single order of **80,995 units**. Because squared errors on 81k units dominate RMSE:

$$\text{Error}^2 \approx (80995 - 27)^2 \approx 6.56 \times 10^9$$

Excluding this single anomalous order (693 test observations):
- **Linear Regression:** **MAE = 242.10 | RMSE = 503.13** (Best)
- **Last-Week Demand:** **MAE = 278.02 | RMSE = 547.58**
- **Random Forest:** **MAE = 283.47 | RMSE = 592.78**
- **Training Average:** **MAE = 334.52 | RMSE = 666.18**

### 5.4 Error Progression Across the 7 Test Weeks
Saved to `artifacts/test_metrics_by_week.csv`:

| Week Start Date | Actual Total Demand | Train Avg MAE | Last Week MAE | Linear Regression MAE | Random Forest MAE | Best Model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2011-10-24** | 45,718 | 347.77 | 279.35 | 282.64 | **278.46** | Random Forest |
| **2011-10-31** | 41,903 | 303.92 | 360.87 | **247.65** | 252.90 | Linear Regression |
| **2011-11-07** | 51,841 | 386.27 | 358.14 | **314.03** | 350.14 | Linear Regression |
| **2011-11-14** | 52,545 | 362.58 | **227.62** | 273.50 | 330.39 | Last Week Demand |
| **2011-11-21** | 41,921 | 283.35 | 241.68 | **187.51** | 237.89 | Linear Regression |
| **2011-11-28** | 47,045 | 330.26 | 244.36 | **211.83** | 292.44 | Linear Regression |
| **2011-12-05** | 119,645 | 1114.05 | 1024.60 | **971.50** | 1033.36 | Linear Regression |

*Takeaway:* Linear Regression achieved lowest MAE on 5 out of the 7 test weeks. *Reference Figure:* `reports/figures/error_by_week.png`.

### 5.5 Per-Product Performance Highlights
- **Model Win Share:** Training Average won 31 items (stable, low-variance products); Linear Regression won 24 items; Random Forest won 23 items; Last-Week won 22 items.
- **Top 5 Easiest Products:** `22969` (MAE: 40.6), `20712` (MAE: 49.9), `20726` (MAE: 56.3), `22384` (MAE: 56.5), `21080` (MAE: 56.8).
- **Top 5 Hardest Products:** `23843` (81k outlier), `23084` (MAE: 2333.2), `22197` (MAE: 1595.2), `84077` (MAE: 843.0), `22086` (MAE: 648.6). *Reference Figure:* `reports/figures/actual_vs_predicted_top_products.png`.

---

## 6. What-If Pricing Tool & Monotonicity Warnings (`src/pricing.py`)

### 6.1 Methodological Principle
The pricing tool evaluates what-if revenue:

$$\text{Simulated Revenue} = P_{\text{cand}} \times \hat{Q}(P_{\text{cand}} \mid X_{\text{lags fixed}})$$

It does **not** assume causal demand response ($\frac{dQ}{dP}$), because historical retail price fluctuations are confounded by promotions and wholesale batch ordering.

### 6.2 Candidate Prices Inside $[P_5, P_{95}]$
For each product that passed the price check, candidate prices were generated strictly inside the empirical 5th to 95th percentiles of historical weekly prices. Non-price features were held fixed at their reference values.

### 6.3 Empirical Monotonicity Check Results (62 Passed Products)
- **Linear Regression:** **0 / 62 warnings**. Because the global coefficient is $\beta_{\text{price}} = -19.67 < 0$, predicted demand drops monotonically with price for all products.
- **Random Forest:** **13 / 62 products triggered rising demand warnings**. Observational decision splits grouped certain higher price tiers with high-demand holiday weeks. For instance, on StockCode `22197` (*SMALL POPCORN HOLDER*), Random Forest predicted demand to rise from 514.7 to 562.5 units (+47.8 units) between £0.90 and £0.94.

This empirical result validates the project's requirement to issue prominent warnings when predicted demand rises with price.

---

## 7. Interactive Streamlit Dashboard (`app/streamlit_app.py`)

The web dashboard exclusively loads precomputed artifacts from `artifacts/`:
- **Instant Launch:** Starts in `<0.5` seconds with no runtime model training.
- **Persistent Warning Banner:** Fixed at the top of the interface declaring non-causal pricing semantics.
- **Interactive Views:**
  1. Product Selector with searchable `StockCode` and `Description`.
  2. Full 54-week historical sales timeline with Train / Val / Test visual partitions.
  3. 7-week out-of-sample actual vs. predicted trajectory charts.
  4. Benchmark metrics and weekly error breakdown tables.
  5. What-if candidate pricing tool ($[P_5, P_{95}]$) with dual charts and dynamic monotonicity alerts.

---

## 8. Verification & Test Suite Summary

A comprehensive pytest test suite of **30 unit tests** was executed:

```powershell
# Run the complete test suite
py -m pytest tests/ -v
```

```
============================== 30 passed in 4.74s ==============================
```

### Complete Reproduction Checklist:

```powershell
# 1. Clean raw transactions (522,537 rows retained)
py src/cleaning.py

# 2. Run EDA and price stability check
py src/price_check.py

# 3. Engineer causal features and perform chronological split
py src/features.py

# 4. Train models and tune Random Forest on validation weeks
py src/models.py

# 5. Evaluate test set strictly once
py src/evaluate.py

# 6. Run pricing what-if simulation inside 5th-95th percentile
py src/pricing.py

# 7. Or execute the entire pipeline end-to-end
py src/run_pipeline.py

# 8. Launch the Streamlit dashboard
py -m streamlit run app/streamlit_app.py
```

---

## 9. Conclusion

The ES 335 ML Project establishes a disciplined, leak-free machine learning workflow for retail demand forecasting:
1. **Methodological Rigor:** Strict chronological splitting (70/15/15) and validation-only tuning prevented lookahead bias.
2. **Model Insights:** Simple, regularized OLS Linear Regression demonstrated superior generalizability (**Test MAE: 355.52**) compared to non-linear tree ensembles on holiday retail demand.
3. **Transparent Economic Semantics:** The what-if pricing tool explicitly exposes the limitations of observational regression, flagging non-causal demand increases and guiding users on sound interpretation.
