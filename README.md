# World Happiness 2015: Predicting Happiness Score

A beginner machine learning project (regression): predicting a country's happiness score from six socio-economic indicators.

## Data
- World Happiness Report 2015 (Kaggle): 158 countries, 12 columns, no missing values.
- Target: `Happiness Score`.
- Features: `Economy (GDP per Capita)`, `Family`, `Health (Life Expectancy)`, `Freedom`, `Trust (Government Corruption)`, `Generosity`.
- Excluded: `Happiness Rank`, `Dystopia Residual`, `Standard Error` (data leakage, see below), and `Country`, `Region` (text columns).

## What I did
1. Loaded and inspected the data with pandas (shape, types, missing values, `describe()`).
2. Exploratory analysis: correlations with the target and scatter plots.
3. Train/test split (80/20, `random_state=42`): 126 and 32 countries.
4. Models: mean Baseline, Linear Regression, Random Forest.
5. Metrics: MAE, RMSE, R².
6. 5-fold cross-validation with shuffling (`KFold`).
7. Feature importance from Random Forest.
8. Demonstration of data leakage.

## Results

Test set (32 countries):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 1.029 | 1.194 | -0.005 |
| Linear Regression | 0.402 | 0.492 | 0.829 |
| Random Forest | 0.432 | 0.522 | 0.808 |

5-fold cross-validation, R²:

| Model | Per fold | Mean |
|---|---|---|
| Linear Regression | 0.829, 0.761, 0.722, 0.684, 0.705 | 0.74 |
| Random Forest | 0.791, 0.751, 0.736, 0.723, 0.618 | 0.724 |

Feature importance (Random Forest): GDP 0.406, Health 0.200, Family 0.194, Freedom 0.099, Trust 0.057, Generosity 0.044.

## Conclusions
- Both models clearly beat the baseline: error of about 0.4 points vs 1.03.
- Linear Regression and Random Forest perform similarly (0.74 vs 0.724 in cross-validation), so the difference is not meaningful on such a small dataset.
- GDP per capita is the feature most strongly related to happiness. This is a correlation, not proof of causation.

## Lessons and limitations
- **Data leakage.** Adding `Dystopia Residual` gives R² = 1.0, because this column plus the six factors makes up the score itself. Such features must not be used.
- **Sorted data and cross-validation.** The file is sorted by score, so `cross_val_score` without shuffling gave negative R² (from -11 to -3). Using `KFold(shuffle=True)` fixed it.
- Small dataset (158 countries), one year only (2015).
- Congo (Kinshasa) has 0 in the feature columns, possibly missing data recorded as zero. The row was kept as is.

## How to run
1. Download `2015.csv` from Kaggle (World Happiness Report).
2. Open `happiness_project.ipynb` in Google Colab.
3. Upload the CSV to Colab and run all cells.

Tools: Python, pandas, NumPy, scikit-learn, matplotlib.
