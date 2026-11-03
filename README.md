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
* Price Volatility Metrics: Derived price variations between December and January for variable and fixed energy tiers (off_peak, peak, mid_peak).Calculated annual mean and maximum monthly price spreads to capture macro price shocks.
  * Temporal & Contract Features:Transformed key calendar dates (date_activ, date_end, date_modif_prod, date_renewal) into duration metrics (e.g., contract tenure in years, months to renewal).
  * Skewness Correction: Applied logarithmic transformation ($\log_{10}(x + 1)$) to heavily right-skewed variables (such as 12-month energy consumption and forecast metrics) to reduce outlier distortion.
  * Categorical Encoding: Applied binary encoding to boolean features (has_gas) and one-hot encoding (pd.get_dummies) to categorical indicators (channel_sales, origin_up), filtering out infrequent categories to prevent matrix sparsity.
 
## 
