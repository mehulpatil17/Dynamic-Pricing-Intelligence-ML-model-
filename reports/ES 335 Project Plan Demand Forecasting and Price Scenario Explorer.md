# ES 335 Project Plan: Demand Forecasting and Price Scenario Explorer

This plan separates **verified facts** (from the UCI dataset page), **recommendations**, and **things to verify** by inspecting the data. No results are claimed; everything here is planned work.

## 1. Recommended final project scope

### Dataset check

| Item | Status | Detail |
| --- | --- | --- |
| Size and period | Verified | 541,909 rows; 1 Dec 2010 to 9 Dec 2011; UK-based non-store online retailer |
| Fields | Verified | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice (sterling), CustomerID, Country |
| Unit cost | Verified absent | No cost field, so profit cannot be computed |
| Cancellations | Verified | InvoiceNo starting with 'C' marks a cancellation |
| Customer mix | Verified | Page says many customers are wholesalers, so bulk orders and price are entangled |
| Format and licence | Verified | One 22.6 MB Online Retail.xlsx; CC BY 4.0 |
| Missing values | Verify | Page says none, but CustomerID and Description are commonly reported as having gaps; check |
| StockCode | Verify | Page says 5-digit; non-product codes (postage, fees) and letter suffixes may exist |
| Negative or zero Quantity/UnitPrice | Verify | Likely linked to cancellations or adjustments |
| Weekly coverage | Verify | About 53 weeks, with partial first and last weeks (1 Dec 2010 and 9 Dec 2011 are mid-week) |
| Price variation per product | Verify | The deciding factor for the pricing part |

The page lists classification and clustering as associated tasks, so demand forecasting is our own framing. That is fine, but we must define the target ourselves.

### Does the data support the research question?

- **Demand forecasting: yes**, with caveats. Product-week aggregation is justified because individual transactions are too noisy (bulk orders, one-off baskets). Only about 53 weeks exist, so the test period will be the pre-Christmas weeks, which training data cannot show (training has only the post-peak Dec 2010 tail). Models may under-predict the peak; we report this rather than hide it.
- **Price optimization: not supported as stated.** Observational prices, no costs, wholesale mixing, and likely few price changes per product cannot support causal optimization.

### Recommended changes

1. **Retitle** to "Demand Forecasting and Price Scenario Exploration for Online Retail" (drop "Intelligence" and "Optimization", which over-promise).
2. **Main contribution:** weekly demand forecasting with honest baselines. **Secondary:** a what-if revenue explorer for only those products that pass a price-variation check.
3. **Limit to about 20 to 50 products** with sales in most weeks, chosen from training weeks only.
4. **Add a last-week baseline** beside the training-average baseline. It is cheap and a stronger reference.
5. **Fallback** if too few products pass: forecasting plus a descriptive price-demand history view.

## 2. Proposal answers

The form wording for field 7 ("Track-specific plan") is not given, so answer 7 assumes it asks how the project fits ES 335 and how the work will be done. Adjust if the form says otherwise. Counts below are exact (checked with code).

### 1. Research question (665 / 1,200)

Using the UCI Online Retail transactions (541,909 rows, 1 Dec 2010 to 9 Dec 2011), can weekly product demand be forecast more accurately by Linear Regression and Random Forest models, using lagged-demand, calendar, and price features, than by naïve baselines (training-period average demand per product and last-week demand) when evaluated on a chronologically held-out period? Secondary question: for products with sufficient price variation, how does the estimated weekly revenue of a trained model change across candidate prices within the observed price range, and how reliable are such estimates given that the data are observational and contain no unit costs?

*Why:* it is testable, names the real dataset and period, separates forecasting from pricing, and states the observational limit upfront.

### 2. Problem and intended claim (1,074 / 1,400)

Retailers set prices without knowing how demand might respond, and demand must first be forecast reliably before any price scenario is meaningful. This project builds a demand-forecasting pipeline for selected products from a UK online gift retailer, plus an interactive tool to explore estimated revenue at candidate prices. The intended claim is limited: on a chronologically held-out test period, a Random Forest and/or Linear Regression model using lagged-demand, calendar, and price features will be compared against a training-average baseline and a last-week baseline using MAE and RMSE; we report whichever result we find, including if the baselines are hard to beat. The pricing tool is a what-if explorer: it multiplies a candidate price by model-predicted quantity (revenue = price × predicted quantity) for products with enough price variation. We do not claim causal price effects, reliable price elasticity, or profit optimization, because the data are observational and contain no unit costs; maximizing predicted revenue is not the same as maximizing profit.

*Why:* the claim is a comparison, not a promised win, and the exclusions are explicit.

### 3. Data plan (1,454 / 1,800)

Data: UCI Online Retail (Chen, 2015; CC BY 4.0), one file (Online Retail.xlsx) with 541,909 transactions from 1 Dec 2010 to 9 Dec 2011. Documented fields: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice (sterling), CustomerID, Country. There is no unit-cost field. Cleaning (rules to be confirmed by inspection): check missing values, duplicates, cancellation invoices (InvoiceNo starting with 'C'), non-positive Quantity or UnitPrice, and non-product StockCodes (e.g., postage or fees); document each rule and the rows removed. Aggregation: weekly units sold per product (StockCode) with weekly mean UnitPrice, for a limited set of products with sales in most weeks, selected using training weeks only; partial first and last weeks are checked and likely excluded. Target: units sold per product-week. Features: lagged demand (1, 2, 4 weeks), rolling mean of past demand (shifted to exclude the current week), week of year, and the week's price. Split: chronological train/validation/test (about 70/15/15% of weeks; exact cut-offs set after inspecting weekly volume). Preprocessing and product selection use training data only. Price check: for each candidate product, count distinct weekly price levels and weeks at each level; only products with enough variation enter the pricing tool (thresholds set after inspection). The data span about 13 months, so seasonality cannot be validated across years; we will state this limitation.

*Why:* uses only documented fields, marks every inspection-dependent decision, and states leakage-safe handling.

### 4. Baseline and candidate methods (1,196 / 1,600)

Baselines: (1) training-period average weekly demand per product; (2) last-week demand (lag-1 naïve), a trivial but stronger reference for time series. Candidates: (3) Linear Regression on lagged-demand, calendar, and price features: interpretable, fast, and shows whether a linear relationship is enough; (4) Random Forest Regression with the same features: captures nonlinear effects with little tuning and provides feature importances. We use only these two models because the project has about four weeks and the goal is a sound, reproducible workflow, not a large model count. Random Forest settings (number of trees, maximum depth, minimum samples per leaf) are tuned on validation weeks only, with a small search. Whether to fit one pooled model across products or per-product models will be decided on validation results; pooled is the default because each product has at most about 53 weekly observations. If demand is highly skewed, a log(1 + units) target will be tested on validation data. Trade-off: Linear Regression is easier to explain; Random Forest may fit nonlinear patterns better but cannot predict beyond demand levels seen in training, which matters near the seasonal peak.

*Why:* a simple-to-complex ladder with a reason for each rung and no filler algorithms.

### 5. Evaluation plan (1,210 / 1,600)

Metrics: MAE (average error in units) and RMSE (penalizes large errors, relevant for bulk orders), reported on the test weeks for all four methods, overall and per product. Splits are chronological: fit on training weeks, tune on validation weeks, evaluate once on the final test weeks; no random shuffling. Leakage controls: lag and rolling features use only earlier weeks; product selection, price-variation checks, and any scaling or imputation use training data only; the test set is never used for tuning. Final models may be refit on train plus validation before the single test evaluation. Interpretation: Linear Regression coefficients, Random Forest feature importances, actual-versus-predicted plots, and residuals by week to see where errors are largest (e.g., the pre-Christmas period). Because the test set has few weeks and products, we will not overclaim small differences. Pricing: causal accuracy cannot be evaluated (no experiment, no costs). We instead check that scenarios stay within the observed price range, inspect how predicted demand changes with price, and flag implausible patterns. Success means a reproducible pipeline and an honest comparison, even if baselines are hard to beat.

*Why:* proper time-based validation, named metrics, explicit leakage rules, and an honest statement of what cannot be evaluated.

### 6. Likely challenge (829 / 900)

The main challenge is that the data may not support the pricing part. Most products probably have few distinct prices, and price changes are confounded with seasonality, promotions, wholesale (bulk) orders, and stock availability, so a price-demand relationship learned here is associational, not causal. There are also no unit costs, so revenue cannot be converted to profit. Mitigation: check price variation per product before modeling; restrict the pricing tool to products with enough variation and to the observed price range; present results as estimated revenue scenarios with clear warnings; and, if too few products qualify, fall back to forecasting plus a descriptive price-demand history view. A second challenge is the short window (about 13 months), which limits seasonal learning and makes the test period unusual.

*Why:* names the real risk, with a mitigation and a fallback. Only 71 characters of headroom, so trim a sentence if you edit.

### 7. Track-specific plan, ES 335 (874 / 1,200)

The project follows the ES 335 workflow end to end: data understanding and cleaning, EDA, feature engineering, baseline selection, model comparison (Linear Regression vs Random Forest), chronological validation, MAE/RMSE evaluation, interpretation, and a Streamlit demo. Plan (four weeks, two students): Week 1: inspect, clean, and explore the data; verify price variation (joint). Week 2: weekly product-level features, baselines, and models (Student A: data and features; Student B: models). Week 3: test-set evaluation and interpretation (A); price-scenario calculation (B). Week 4: Streamlit dashboard (A); tests, README, and report (B); presentation (joint). Deliverables: reproducible code with requirements file, short report, working dashboard, and a limitations section. Fallback: if price variation is inadequate, deliver forecasting plus a descriptive price view.

*Why:* maps the work to course topics and gives a division of labour. Replace "Student A/B" with your names.

## 3. Detailed ML methodology

### Pipeline

| Step | Decision | Status |
| --- | --- | --- |
| Clean | Drop cancellations (InvoiceNo starts with 'C'), non-positive Quantity/UnitPrice, non-product StockCodes; log row counts per rule | Verify rules on data |
| Aggregate | Product-week: sum of Quantity, mean UnitPrice, week start date | Justified: transaction-level demand is too noisy |
| Select products | Sales in most training weeks; top by training volume; about 20 to 50 | Choose from training weeks only |
| Target | Units sold per product-week (optionally log(1 + units)) | Test log on validation |
| Features | lag 1, 2, 4; rolling mean of lags (shifted); week of year; week's mean price; price relative to the product's training median | Do not use same-week invoice or customer counts: they leak |
| Split | Chronological about 70/15/15 of weeks (roughly 36/8/8) | Set cut-offs after viewing weekly volume |
| Fit and tune | Tune Random Forest on validation weeks only | Keep search small |
| Final test | Evaluate once on test weeks | Never retune after |

### Model trade-offs

| Model | Interpretability | Accuracy potential | Time cost | Main risk |
| --- | --- | --- | --- | --- |
| Training-average baseline | Very high | Low | Minutes | Ignores trends |
| Last-week baseline | Very high | Often strong for weekly data | Minutes | Noisy if demand is spiky |
| Linear Regression | High (coefficients) | Moderate | Hours | Misses nonlinear and product effects |
| Random Forest | Medium (importances) | Moderate to high | Hours to a day | Cannot extrapolate beyond training range |

We do not add gradient boosting, neural nets, or ARIMA. They would add tuning time and weaken interpretability without adding new course concepts. If time remains in Week 4, a single optional extra (such as a gradient boosting comparison) is acceptable, but only after the core is done.

### Leakage and validation rules

- Compute every lag and rolling feature with a one-week shift so week *t* never sees week *t* or later.
- Fit any scaler, encoder, or imputer on training weeks only.
- Choose products and price-variation thresholds from training weeks only.
- Touch the test weeks once.
- Optional: expanding-window validation folds inside the training+validation period for a steadier model choice.
- Report MAE and RMSE overall and per product; RMSE will be dominated by large bulk orders, which is informative, not a bug.
- Expect the test period to contain the pre-Christmas rise. Report the error by week and say plainly if models under-predict.

## 4. Pricing simulation methodology and limitations

### Prediction is not causal estimation

Our model answers: "given recent demand, the calendar, and this week's price, what quantity do we expect?" It does not answer: "what would happen to demand if we changed the price?" In historical data, prices moved for reasons we cannot see (promotions, wholesale deals, stock clearance, seasonality), and those same reasons move demand. So a price coefficient may reflect them, not a price effect. Only a randomized price test or a carefully justified causal design could support causal claims, and this dataset offers neither.

### Steps

1. **Check price variation** per candidate product, using training weeks: number of distinct weekly price levels, weeks at each level, min/max ratio and coefficient of variation of weekly price. Starting heuristic (adjust after inspection): at least 3 distinct price levels, each seen in several weeks. Mark the thresholds as *to verify*.
2. **Select products** that pass the check and have steady sales. Report how many pass. If very few do, use the fallback below.
3. **Define candidate prices** inside the product's observed range (for example, 5th to 95th percentile of weekly price). Never go outside it.
4. **Simulate:** for a chosen future week, hold the lag and calendar features fixed, vary only the price feature, and predict quantity.
5. **Compute estimated revenue** = candidate price × predicted quantity.
6. **Sanity-check:** if predicted demand rises as price rises, show a warning. It likely reflects confounding, not reality.

### Why maximum predicted revenue is not maximum profit

Profit = (price − unit cost) × quantity. A price that maximizes revenue can have a worse margin, because a lower price with higher volume may sell below a healthy margin, or a higher price may shift volume to costlier handling. Without unit cost, profit and the profit-maximizing price are unknowable from this dataset.

### Handling missing cost

- Core demo: show estimated **revenue only**, labeled as such.
- Optional: let the user type an **assumed** unit cost to see illustrative gross margin, labeled "hypothetical, not from data".
- Never present a "best price" as a recommendation. Use wording like "highest estimated revenue among the scenarios shown".

### Elasticity

Do not report price elasticity as a finding. If you want an exploratory view, a log-log regression on products that pass the check is acceptable, with confidence intervals and an "associational, not causal" label. Skip it if it threatens the schedule.

### Honest fallback

If the check shows few products with real price variation, replace scenario simulation with a **descriptive price-demand history view** (weekly price and demand plotted together, price-band comparison of average weekly units) plus the forecasting results. This is a legitimate, defensible outcome and should be said so in the report.

## 5. Website features and user workflow

### Minimum viable demo (Streamlit)

| Page or section | Content |
| --- | --- |
| Product selector | Dropdown of products used in the project, showing StockCode and Description |
| History | Weekly demand and weekly price charts for the chosen product |
| Actual vs predicted | Test-period line chart: actual, baseline, Linear Regression, Random Forest |
| Model comparison | Table of MAE and RMSE for all four methods (overall, plus the chosen product) |
| Price scenarios | Slider or list of candidate prices within the observed range; table and bar chart of estimated revenue |
| Warnings | Always-visible banner: associational model, no unit costs, revenue is not profit, one year of data |

**User workflow:** pick a product, view its history, check how well each model forecast the test weeks, open price scenarios, read estimated revenue for each price with the warnings visible. Products that fail the price-variation check show "insufficient price variation for scenarios" and hide the revenue chart.

### Optional improvements (only after the core works)

- Assumed-cost input for hypothetical margin
- Prediction intervals or error bands
- Compare two products side by side
- Download scenario table as CSV
- Feature-importance chart for the chosen model

Precompute predictions and metrics in the pipeline and save to files, so the app only loads and plots. This keeps the demo fast and removes risk of live training failures.

## 6. Folder structure and tools

See Annexure A for the tree. Purposes:

| Component | Purpose |
| --- | --- |
| data/raw, data/processed | Original xlsx (not edited) and cleaned and weekly tables |
| notebooks/ | Inspection, EDA, and exploratory checks; numbered in order |
| src/ | Reusable functions: cleaning, features, models, evaluation, pricing |
| app/ | Streamlit app that reads precomputed outputs |
| tests/ | A few pytest checks (for example, lag features never use the current week; split order is chronological) |
| reports/ | Figures, metrics tables, final report |
| requirements.txt, README.md | Reproducibility and run instructions |

**Libraries:** pandas, numpy, openpyxl (read the xlsx), scikit-learn, matplotlib or plotly, streamlit, joblib, pytest. Optional: ucimlrepo to fetch the data.

## 7. Four-week timeline

| Week | Student A | Student B | Dependencies | Deliverables |
| --- | --- | --- | --- | --- |
| 1 | Load data, document fields, missing values, duplicates, cancellations | EDA: demand over time, top products, price distributions, partial weeks | Both finish inspection before cleaning rules are frozen | Data dictionary, cleaning rules and row counts, EDA notebook, product shortlist |
| 2 | Product-week aggregation, lag and calendar features, split | Baselines, Linear Regression, Random Forest; validation tuning | B needs A's feature table (aim for mid-week) | Feature table, baseline and model code, validation results |
| 3 | Test-set evaluation, error analysis, interpretation | Price-variation check and scenario calculation | Final models frozen before test evaluation | Metrics table, plots, product list that passes the price check, scenario function |
| 4 | Streamlit dashboard | Tests, README, report, fallback if needed | App needs saved predictions and scenario outputs | Working demo, reproducible repo, report, slides |

**Week 1 tip:** do the price-variation check early (even a rough one), since it decides whether the pricing part is full or fallback. Both students should run the whole pipeline once by the end of Week 2, to catch integration problems early. Use Git with small commits and agree on file names for saved outputs on day one.

## 8. Risks, fallback plans, and submission checklist

| Risk | Likelihood | Fallback |
| --- | --- | --- |
| Few products with price variation | High | Descriptive price view plus forecasting only |
| Models do not beat last-week baseline | Medium | Report honestly; discuss why (short history, noisy demand) |
| Test period peak not seen in training | High | Report errors by week; explain tree extrapolation limit |
| Data cleaning takes longer than planned | Medium | Freeze simple rules in Week 1; log exclusions rather than perfecting them |
| Dashboard slips | Medium | Minimum viable demo first; static report figures as backup |
| Team integration problems | Medium | Shared file contracts and a mid-Week-2 end-to-end run |

### Final checklist

- [ ] Dataset fields and counts in the report match the UCI page, and every "to verify" item is resolved
- [ ] Cleaning rules and removed-row counts documented
- [ ] Chronological split; test set used once
- [ ] Both baselines and both candidate models evaluated with MAE and RMSE
- [ ] No leakage (tests pass)
- [ ] Price-variation check results reported, including how many products pass
- [ ] Pricing wording avoids "optimal", "causal", "elasticity", and "profit" unless supported
- [ ] Dashboard shows warnings and runs from README steps
- [ ] Proposal claims match what was actually done
- [ ] Dataset cited (Chen, 2015, doi:10.24432/C5BW33)

## Annexure A: folder structure

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
