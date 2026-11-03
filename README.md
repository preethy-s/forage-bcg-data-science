# Customer Churn Prediction (BCG Forage Data Science Program)

## Project Overview
PowerCo, a major utilities provider in the SME energy market, is experiencing increasing customer churn. 
To mitigate revenue loss and protect market share, BCG was engaged to analyze customer behavior and determine whether price sensitivity is the primary driver of customer churn. 
This repository contains the end-to-end data science pipeline built to test this hypothesis, engineer predictive behavioral features, train a Random Forest Classifier and deliver actionable business strategies for PowerCo's SME leadership team.

## Key Objectives and Hypotheses
* Test Price Sensitivity Hypothesis: Determine if customer churn is strongly correlated with price changes across peak, off-peak, and mid-peak pricing tiers.
* Feature Engineering: Extract meaningful indicators from raw consumption and pricing history datasets.
* Predictive Modeling: Build and optimize a machine learning model capable of identifying at-risk SME customers prior to contract expiration.
* Business Strategy & Executive Recommendations: Translate model outputs into a targeted retention program that maximizes revenue preservation while minimizing unnecessary discount expenditures.

## Feature Engineering Workflow
To transform raw historical customer and transaction logs into predictive features, the following pipeline was executed:
* Price Volatility Metrics: Derived price variations between December and January for variable and fixed energy tiers (off_peak, peak, mid_peak). Calculated annual mean and maximum monthly price spreads to capture macro price shocks.
  * Temporal & Contract Features:Transformed key calendar dates (date_activ, date_end, date_modif_prod, date_renewal) into duration metrics (e.g., contract tenure in years, months to renewal).
  * Skewness Correction: Applied logarithmic transformation ($\log_{10}(x + 1)$) to heavily right-skewed variables (such as 12-month energy consumption and forecast metrics) to reduce outlier distortion.
  * Categorical Encoding: Applied binary encoding to boolean features (has_gas) and one-hot encoding to categorical indicators (channel_sales, origin_up), filtering out infrequent categories to prevent matrix sparsity.
 
## Model Training and Key Findings
A Random Forest classifier is trained on the engineered feature set (61 features) to predict customer churn. The dataset is split 75/25 into train and test sets, and the model is evaluated on the held-out test set to simulate real-world generalization.
The model is a `RandomForestClassifier` with the following configuration:
| Parameter | Value |
|---|---|
| `n_estimators` | 1000 |
| Decision threshold | 0.5 (default) |

A Random Forest was chosen because it handles non-linear relationships well, does not require feature scaling, and provides built-in feature importance, which is valuable for understanding *why* customers churn.

**Evaluation:**
Because the dataset is imbalanced (~10% churners), overall accuracy is misleading, a model that predicts "no churn" for everyone would still score ~90%. 
The evaluation therefore focuses on the **confusion matrix** and **class-1 (churner) precision, recall, and F1** rather than accuracy alone.
The model is very good at identifying customers who will **not** churn, but poor at identifying those who **will**. Of the 366 actual churners in the test set, only 18 were correctly flagged. The current feature set is not discriminative enough to cleanly separate churners from non-churners, a limitation the notebook states honestly rather than overstating model quality.

**Feature Importance:**
Feature importance scores from the trained forest rank every feature by how much it contributes to the model's decision splits. The top drivers are:
- **Net margin** and **12-month consumption** — the strongest signals of churn risk.
- **Margin on power subscription** — a key secondary driver.

This directly informs the original BCG hypothesis: **"Is churn driven by the customer's price sensitivity?"** The evidence suggests **no**, price sensitivity is a secondary factor at best. A conclusive answer would require further experimentation.

## Key Findings

1. **Accuracy is misleading for this problem:** With ~10% churners, a "predict no-churn for everyone" model scores ~90% accuracy and catches zero churners. Precision, recall, and F1 for class 1 are the metrics that matter.

2. **The model identifies non-churners well but churners poorly:** Recall of ~0.05 means only 18 of 366 churners were correctly flagged. This is a known limitation of training on imbalanced data without class weighting or threshold adjustment.

3. **Top churn drivers:** are net margin, 12-month consumption, power-subscription margin, and tenure-related features (months active, months since contract update).

4. **Price sensitivity is not a primary driver of churn:** it appears only as a weak contributor. This partially answers the original BCG hypothesis with **no**, though further experimentation would be needed for a conclusive result.

5. **The engineered features alone are not strongly discriminative** for churn: The notebook documents this transparently rather than overstating the model's capabilities.


## Acknowledgments
Boston Consulting Group (BCG) X and Forage for providing the dataset, virtual experience workspace, and guidance templates.

## 👤 Author
* **Preethy S**
