# Portfolio Optimisation Strategies
## Overview
Showcased machine learning in portfolio construction by training/testing multiple optimisation strategies (Max Diversification, Min CVaR, Equal Weighted) on a portfolio of ETFs, with results deployed in a Python Vizro dashboard.
Deployed on HuggingFace Spaces.

## Python Tools Used

### Data Fetching, Preparation and Plotting
* **yfinance** to obtain historical ETF prices
* **pandas** for data processing and preparation
* **plotly** for plotting the data

### Portfolio Optimisation
* **scikit-learn** for training and testing
* **skfolio** to access the Maximum Diversification and Minimum CVaR models

### Results Dashboard
* **vizro** to build the dashboard

## Description
### Portfolio Composition
Five ETFs:
- **EWA** iShares MSCI Australia ETF
- **EWH** iShares MSCI Hong Kong ETF
- **EWJ** iShares MSCI Japan ETF
- **ENZL** iShares MSCI New Zealand ETF
- **EWS** iShares MSCI Singapore ETF

### Time Period
30/09/2010 to 30/09/2024

### Optimisation Models Tested:
- Maximum Diversification
- Minimum CVaR
- Equal Weighted

### Metrics
- Annualised Mean
- Annualised Standard Deviation
- CVaR at 95%

## Results
- The number of stocks (ETFs) in the portfolio is very small, at only 5, so there was not much variability in the results.
- The Equal Weighted model had the highest annualised return and annualised standard deviation. 

## Next Steps
Test the optimisation models on a portfolio with a greater number of stocks, over different time periods, including other optimisation models in the analysis.

## Link to HuggingFace Spaces
(https://huggingface.co/spaces/Shantala/VizroDashboardPortfolioOptimisation)
<img width="1897" height="890" alt="image" src="https://github.com/user-attachments/assets/89f8fbae-4dfa-454f-8101-f7594a312acf" />
