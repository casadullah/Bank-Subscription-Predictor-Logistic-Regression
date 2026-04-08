# Bank Subscription Predictor — Logistic Regression

## What this project does
Predicts whether a bank customer will subscribe to a term 
deposit based on call characteristics and market conditions.
Built using Binary Logistic Regression with Statsmodels on 
the Bank Marketing dataset (518 training + 222 test records).

## Concepts demonstrated
- Simple Logistic Regression (single variable)
- Multiple Logistic Regression (5 variables)
- Odds ratio calculation using np.exp()
- Pseudo R-squared interpretation
- Custom confusion matrix function built from scratch
- Train vs test accuracy comparison
- Binary outcome encoding (yes/no → 1/0)

## Key findings
- Duration is the strongest positive predictor — 
  every additional second multiplies subscription 
  odds by 1.005 (compounding effect)
- High interest rates strongly reduce subscription 
  probability (coef: -0.800)
- March is a significantly bad month for conversions 
  (coef: -1.832)
- Previous campaign contact increases subscription 
  likelihood (coef: 1.536)
- Pseudo R² improved from 0.21 (simple) to 0.51 
  (multiple) — adding variables doubled model power

## Model performance
| Metric | Value |
|--------|-------|
| Pseudo R² (simple) | 0.2121 |
| Pseudo R² (multiple) | 0.5140 |
| Training accuracy | 86.3% |
| Test accuracy | 86.0% |
| Generalization gap | 0.3% — no overfitting |

## Tools used
Python · Pandas · NumPy · Statsmodels · Seaborn · SciPy

## How to run
pip install pandas numpy statsmodels seaborn scipy
jupyter notebook Logistic_Regression_project.ipynb
