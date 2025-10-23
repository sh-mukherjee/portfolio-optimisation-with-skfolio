# Import the following for the Vizro dashboard
from vizro import Vizro
import vizro.plotly.express as px
import vizro.models as vm
from vizro.models.types import capture
from vizro.tables import dash_data_table

# Import the following for financial data, data preparation and plotting
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Import the following for portfolio optimisation
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import EqualWeighted, MaximumDiversification, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

# The following are to ensure that Colab can run the dashboard
#import dash._callback_context
#dash._callback_context.context_value.set({})
#Vizro._reset()

# DATA DOWNLOAD AND PREPARATION

# List of tickers for iShares MSCI ETFs Australia, Hong Kong, Japan, New Zealand, Singapore (countries in MSCI Pacific index, which includes large and mid cap equities in these countries)
tickers = ['EWA', 'EWH', 'EWJ', 'ENZL', 'EWS']

# Ticker for the iShares MSCI Pacific IMI ETF (this includes large, mid and small cap equities in the five DM APAC countries mentioned above)
mkt_idx = 'IPAC'

# Download historical data (daily) for the tickers and market index ETF
data = yf.download(tickers, start="2010-09-30", end="2024-09-30", auto_adjust=False)
idx_data = yf.download(mkt_idx, start="2010-09-30", end="2024-09-30", auto_adjust=False)

# Extract the closing prices (adjusted for splits and dividends)

prices = pd.DataFrame(data['Adj Close'])
idx_prices = pd.DataFrame(idx_data['Adj Close'])

# Format the index to display only the date
#prices.index = prices.index.strftime('%Y-%m-%d')
#idx_prices.index = idx_prices.index.strftime('%Y-%m-%d')

# Obtain the series of daily returns
X = prices_to_returns(prices)
IDX = prices_to_returns(idx_prices)

# OPTIMISATION MODEL
# Split the data into training and testing data sets, with a training-testing split of 0.65-0.35
X_train, X_test = train_test_split(X, test_size=0.35, shuffle=False)

# Grab the market index ETF daily return series for the same period as the testing data set
IDX_test = IDX.loc[X_test.index[0]:]

# We will also create a joint dataframe of daily returns and cumulative growth for the five portfolio assets and the IPAC ETF
# daily_prices = pd.concat([prices,idx_prices], axis=1)
# daily_prices.columns = tickers + [mkt_idx]

daily_ret = pd.concat([X_test, IDX_test], axis=1)
daily_ret.columns = tickers + [mkt_idx]
cumulative_growth = (1+daily_ret).cumprod()*10000

# OPTIMISATION MODELS

# Model 1: Maximum Diversification -- create the model and fit it on the training data set
model1 = MaximumDiversification()
model1.fit(X_train)

# Model 2: Minimum CVaR -- create the model and fit it on the training data set
model2 = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    portfolio_params=dict(name="Min CVaR"),
)
model2.fit(X_train)

# Comparator: Equal Weighted Portfolio -- create the model and fit it on the training data set
eq = EqualWeighted()
eq.fit(X_train)

# Next, we will create a dataframe to show the weights of the assets in each of these models
# Create a dictionary for the models
data = {'Maximum Diversification': model1.weights_, 'Minimum CVaR': model2.weights_, 'Equal Weighted': eq.weights_}

# Create a DataFrame
df_weights = pd.DataFrame(data)

# Insert a new column with the tickers at the beginning (index 0)
df_weights.insert(0, 'Asset', tickers)

# Reshape the data for the Plotly chart (melt the dataframe)
df_weights_melted = df_weights.melt(id_vars='Asset', var_name='Portfolio', value_name='Weight')

# Define a function to plot the stacked bar chart for the portfolio weights, and use 'capture' decorator to refer to the plotting function in the Vizro dashboard
@capture("graph")
def composition(data_frame, x, y, color=None):
    fig0 = px.bar(data_frame=data_frame, x=x, y=y, color=color)
    fig0.update_layout(
        xaxis_title = "Portfolio",
        yaxis = {
                "title": "Asset Weight",
                "tickformat": ",.0%",
            },
        legend_title_text = "Asset"
    )
    return fig0

# Analyse the results of the model on the training data set
"""
ptf_model1_train = model1.predict(X_train)
ptf_model2_train = model2.predict(X_train)
ptf_eq_train = eq.predict(X_train)
"""

# Predict the model and the comparator on the test data set
ptf_model1_test = model1.predict(X_test)
ptf_model2_test = model2.predict(X_test)
ptf_eq_test = eq.predict(X_test)

# Daily return series of the models and comparator
df_returns = pd.concat([ptf_model1_test.returns_df, ptf_model2_test.returns_df, ptf_eq_test.returns_df], axis=1)
df_returns.columns = ['Maximum Diversification', 'Minimum CVaR', 'Equal Weighted']

# Cumulative returns of the models and comparator
#df_cumul_growth = (1+df_returns).cumprod()-1
df_cumul_growth = (1+df_returns).cumprod()*10000

# Calculate cumulative returns of the market index ETF from the 'IPAC' column
#idx_cumulative_returns = (1 + IDX_test['IPAC']).cumprod() - 1
idx_cumulative_growth = (1 + IDX_test['IPAC']).cumprod()*10000

# Join these cumulative returns into one data frame
df_cumul_ret = pd.concat([df_cumul_growth, idx_cumulative_growth], axis=1)

df_cumul_ret = df_cumul_ret.rename(columns={'IPAC':'iShares Core MSCI Pacific ETF'})


# Define a function to plot the line chart for the cumulative returns, and use 'capture' decorator to refer to the plotting function in the Vizro dashboard
@capture("graph")
def cumulative(data_frame, x, y):
    fig1 = px.line(data_frame=data_frame, x=x, y=y)
    fig1.update_layout(
    xaxis_title = None,
    yaxis = {
                "title": None,
                #"tickformat": ",.0%",
            },
    legend_title_text = "Portfolios"
    )
    return fig1

# For improved analysis, it's possible to load all predicted portfolios into a :class:~skfolio.population.Population:
population = Population([ptf_model1_test, ptf_model2_test, ptf_eq_test])

# Select the summary stats to display in a data table
summ_df = population.summary().loc[['Mean','Annualized Mean', 'Standard Deviation', 'Annualized Standard Deviation', 'CVaR at 95%', 'Value at Risk at 95%', 'MAX Drawdown', 'Effective Number of Assets', 'Assets Number']].reset_index()
summ_df.rename(columns={'index': 'Measure'}, inplace=True)

# CREATE THE DASHBOARD

first_page = vm.Page(
    title="Portfolio Assets and Optimisation Models",
    layout=vm.Layout(grid=[[0, 1, 2], [3, 3, 3]]),
    components=[
        vm.Card(
            text="""
            # Portfolio Assets

            Our portfolio holds 5 ETFs that track the MSCI indices of the 5 developed market countries in the MSCI Pacific index.

            These are:

            - **EWA** [iShares MSCI Australia ETF](https://www.ishares.com/us/products/239607/)
            - **EWH** [iShares MSCI Hong Kong ETF](https://www.ishares.com/us/products/239657/)
            - **EWJ** [iShares MSCI Japan ETF](https://www.ishares.com/us/products/239665/ishares-msci-japan-etf)
            - **ENZL** [iShares MSCI New Zealand ETF](https://www.ishares.com/us/products/239672/)
            - **EWS** [iShares MSCI Singapore ETF](https://www.ishares.com/us/products/239678/)

            Each of these ETFs contains large and mid-cap equities of the respective country


            We **do not** have an ETF that tracks the MSCI Pacific Index (large and mid-cap equities), hence we will use the following ETF:
            
            **IPAC** iShares Core MSCI Pacific ETF 
            
            This ETF tracks the MSCI Pacific Investable Market Index, which includes large, mid **and** small cap equities in the five countries mentioned above.
            
            
            How do the optimised portfolios compare against the market index in terms of cumulative growth? Check out the second page. 

            **Note**: the results will be influenced by the concentrated nature of the portfolio, as it tends to magnify the impact of specific assets.
            

            """,
        ),
        vm.Card(
            text = """
            # Python Tools Used
            * **yfinance** to obtain the time series of daily prices of the ETFs from the period 30/09/2010 to 30/09/2024

            * **scikit-learn** to split the data into training and testing data sets with a training-testing split of 0.65-0.35

            * **skfolio** to train and test the Maximum Diversification, Minimum CVaR and Equal Weighted optimisation models

            * **vizro** to build this dashboard
            """
        ),
        vm.Card(
            text = """
            # Optimisation Models

            ## Maximum Diversification

            [Example from skfolio](https://skfolio.org/auto_examples/3_maxiumum_diversification/plot_1_maximum_divesification.html#sphx-glr-auto-examples-3-maxiumum-diversification-plot-1-maximum-divesification-py)
            
            * Find the portfolio that maximises the diversification ratio, which is the ratio of the weighted volatilities over the total volatility.

            * Compare the portfolio composition, cumulative returns and risk statistics with that of the equal weighted portfolio.

            ## Minimum CVaR

            [Example from skfolio](https://skfolio.org/auto_examples/1_mean_risk/plot_2_minimum_CVaR.html#sphx-glr-auto-examples-1-mean-risk-plot-2-minimum-cvar-py)

            * Find the portfolio that minimises CVaR (Conditional Value at Risk) with a default confidence level of 95%.

            * Compare the portfolio composition, cumulative returns and risk statistics with that of the equal weighted portfolio.

            """
        ),
        vm.Graph(title="Cumulative Growth of $10,000 of Portfolio Assets and the iShares Core MSCI Pacific ETF over the testing dataset period",
            id="growth",
            figure=px.line(data_frame=cumulative_growth, x=cumulative_growth.index, y=cumulative_growth.columns.to_list(), labels={
        'X': "",
        'value': "",
        'variable': 'Ticker'
         }
          )
            ),
        
    ],
)

second_page = vm.Page(
    title="Portfolio Composition, Returns and Stats",
    layout=vm.Layout(grid=[[0, 1], [2,2]]),
    components=[
        vm.Graph(title="Portfolio Composition",
            id="portfolio_comparison",
            figure=composition(x='Portfolio', y='Weight', color='Asset', data_frame = df_weights_melted)
            ),
        vm.Table(
            title="Summary Stats",
            figure=dash_data_table(data_frame=summ_df)
        ),
        vm.Graph(title="Cumulative Growth of $10,000 invested in the portfolios and iShares Core MSCI Pacific ETF over the testing dataset period",
            id="cumulative_returns",
            figure=cumulative(x=df_cumul_ret.index, y=df_cumul_ret.columns.to_list(), data_frame = df_cumul_ret)
            ),
    ],
)

dashboard = vm.Dashboard(pages=[first_page, second_page])
app = Vizro().build(dashboard)
server = app.dash.server

if __name__ == "__main__":  
    app.run()
