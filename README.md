Customer Churn Prediction — Telco Dataset

This project came out of my SIP (Summer Internship Programme) work, where I had to figure out why telecom customers leave — and more importantly, which ones are about to.

The dataset is IBM's Telco Customer Churn dataset. It has about 7,000 customer records with details like contract type, monthly charges, tenure, and whether they churned or not.

---
What's inside

-customer-churn-analysis/
├── notebooks/
│   └── cbsot_churn_prediction.ipynb
├── README.md

---
What I did, step by step

1. Exploratory Data Analysis
Started by looking at the basics — tenure distribution, monthly charges, contract types. The boxplots made it pretty clear that customers who churn tend to have higher monthly bills and shorter tenures. Not shocking, but good to confirm before modelling.
 2. Data Cleaning
`Total Charges` had some null values hiding as empty strings. Converted to numeric, filled nulls with 0. Dropped columns that would either leak the target or weren't useful for prediction (CustomerID, city/state/zip, Churn Score, Churn Reason, CLTV).

3. Encoding
Used `pd.get_dummies` with `drop_first=True` to handle all the categorical columns (Internet Service, Contract, Payment Method, etc.).

4. Modelling

Baseline — Random Forest
Started simple: 100 trees, max depth 3. Accuracy was decent but recall on churners was low. The dataset is imbalanced (~73% No, ~27% Yes), so accuracy alone was misleading.

Approach 1 — Balanced class weights
Added `class_weight='balanced'` to the Random Forest. Recall improved significantly on the churn class.

Approach 2 — Hyperparameter tuning
Bumped trees to 300 and depth to 10. Ran a grid over `n_estimators = [100, 200, 300, 400, 500]` and `max_depth = [5, 10, 15, 20]`, sorting results by recall then accuracy. This helped identify the sweet spot.

Approach 3 — Feature importance
Pulled feature importances from the tuned model. Dropped two near-zero features (`Phone Service_Yes`, `Multiple Lines_No phone service`) and retrained. Marginal improvement, but cleaner model.

Approach 4 — XGBoost
Switched to XGBoost with `scale_pos_weight` to handle class imbalance natively. Also used 5-fold cross-validation to get a more honest picture of performance.

5. Customer Segmentation
Used the churn probabilities from the tuned RF model as a feature, combined with tenure, monthly charges, and total charges. Applied KMeans (k=3, chosen via elbow method) and got three segments:

| Segment | Description |
|---|---|
| Budget Loyal Customers | Long tenure, low charges, low churn risk |
| High Risk New Customers | Short tenure, moderate charges, high churn probability |
| Loyal Premium Customers | Long tenure, high charges, low churn risk |

Visualised each segment's churn probability against monthly charges, tenure, and total charges.

---

Results

| Model | Accuracy | Recall (Churn) | ROC AUC |
|---|---|---|---|
| Random Forest (baseline) | ~0.79 | low | — |
| Random Forest (balanced + tuned) | ~0.79 | improved | ~0.84 |
| XGBoost | ~0.79 | best | ~0.85+ |

XGBoost edged out Random Forest on both recall and AUC. Given the business context (it's worse to miss a churner than to flag a non-churner), recall on the positive class was the key metric.

---

Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```

Then open the notebook:

```bash
jupyter notebook notebooks/cbsot_churn_prediction.ipynb
```

The dataset path in the notebook points to `/content/Telco_customer_churn.xlsx` (Google Colab). If you're running locally, update that line to wherever you've placed the file.

---

Notes

- The dataset isn't included in the repo (just add it to `/data/` and update the path).
- Some cells are exploratory / iterative — the notebook isn't fully linear, a few approaches were tested in sequence.
- The segmentation part assumes you've run the full modelling section first, since it uses `rf_tuning`'s probabilities.

---

Tech used

- Python 3
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (Random Forest, KMeans, StandardScaler, cross-validation)
- XGBoost

---


