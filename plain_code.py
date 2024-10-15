











import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from cvxpy import Variable, quad_form, Problem, Minimize, Maximize
from datetime import datetime


# -------------- Dashboard -------------------------

# Question 1:    Stocks or ETFs?
investment_type = 'stocks'           # Options: ['stocks' or 'ETF']

# Window to benchmark: [2 or 5]
time_window = 5

# optimization type [`minimize portfolio risk`, `maximize portfolio return`]:
optimization_type = "minimize portfolio risk"

#         ------ ETFs Questions ------
risk_level_etf = 'high'            # Options: ['low', 'medium', 'high']
#         ------ End ------


#         ------ Stocks Questions ------
risk_level_stocks = 'medium'                # Options: ['low', 'medium', 'high']
# Question 2b [Stocks]: What kind of market capatilization?
size_preference = 'medium'         # Options: ['small', 'medium', 'large']
# Question 2c [Stocks]: Do you want to focus on 'value' stocks or 'growth' stocks in your portfolio?
# value:    Suitable for conservative investors looking for steady returns and lower risk.
# growth:
pe_preference = 'growth'         # Options: ['value', 'growth']
#         ------ End ------


# General piece, how much % is minimal stake and what is your buy in?
percentage_threshold = 10.0
buy_in_EUR = 100000
fee = 1500



















































if investment_type == 'stocks':
  tickers = [
      'AAPL', 'MSFT', 'GOOGL', 'AMZN',
      'TSLA', 'NVDA', 'NFLX', 'ADBE',
      'PYPL', 'INTC', 'CSCO', 'ORCL',
      'IBM', 'CRM', 'AMD', 'QCOM',
      'TXN', 'AVGO', 'MU', 'AMAT',
      'LRCX', 'KLAC', 'ASML', 'TSM',
      'BABA', 'JD', 'PDD', 'BIDU',
      'NTES', 'TCEHY', 'SHOP', 'SQ',
      'SNAP', 'UBER', 'LYFT', 'ZM',
      'DOCU', 'ROKU', 'SPOT', 'ETSY',
      'PINS', 'WMT', 'DIS', 'V', 'MA',
      'JPM', 'BAC', 'WFC', 'C', 'GS',
      'MS', 'BLK', 'T', 'VZ', 'TMUS',
      'CMCSA', 'NFLX', 'CHTR', 'SIRI',
      'NKE', 'ADDYY', 'LULU', 'UAA',
      'UA', 'GPS', 'ANF', 'AEO', 'URBN',
      'ROST', 'TJX', 'KSS', 'M', 'JWN',
      'WMT', 'TGT', 'COST', 'DG', 'DLTR',
      'FIVE', 'BIG', 'OLLI', 'BBY', 'HD',
      'LOW', 'TSCO', 'WSM', 'RH', 'W',
      'AMZN', 'EBAY', 'ETSY', 'W', 'GRPN',
      'PETS', 'CVS', 'WBA', 'CI', 'UNH',
      'HUM', 'CNC', 'MOH', 'HCA', 'UHS',
      'THC', 'CYH', 'ACHC', 'SEM', 'HCSG',
      'EHC', 'ADUS', 'AMED', 'CHE', 'BKD',
      'NHC', 'ENSG', 'GEN', 'NHI', 'OHI',
      'SBRA', 'VTR', 'WELL', 'HCP', 'DOC',
      'HR', 'GMRE', 'MPW', 'CHCT', 'CTRE',
      'SNH',
  ]
elif investment_type == 'ETF':
  tickers = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'VTI',
    'IWM', 'VUG', 'VTV', 'EEM', 'EFA',
    'VEA', 'VWO', 'GLD', 'TLT', 'VYM',
    'FTEC', 'XLK', 'VGT', 'SOXX', 'SMH',
    'XLV', 'VHT', 'IBB', 'XBI',
    'XLF', 'VFH', 'IYF',
    'XLY', 'VCR', 'FDIS',
    'XLE', 'VDE', 'IYE',
    'VEU', 'IXUS'
]
else:
  print(f"warning, wrong input")

# Fetch historical data
data = yf.download(tickers, start=((datetime.now() - pd.DateOffset(years=time_window)).strftime('%Y-%m-%d')), end=datetime.now().strftime('%Y-%m-%d'))['Adj Close']


if investment_type == 'stocks':
  # Function to retrieve metadata on the stocks
  def get_metadata(ticker, meta_type='marketCap'):
      stock = yf.Ticker(ticker)
      return stock.info.get(meta_type, None)

  def retrieve_and_filter_metadata(tickers, risk_level_stocks):
      market_cap = {}
      beta = {}
      pe_ratio = {}
      excluded_tickers = []

      # Define P/E ratio boundaries based on risk level
      if risk_level_stocks == 'low':
          pe_min, pe_max = 5, 50
      elif risk_level_stocks == 'medium':
          pe_min, pe_max = 0, 100
      elif risk_level_stocks == 'high':
          pe_min, pe_max = -100, 100
      else:
          raise ValueError("Invalid risk level. Choose from ['low', 'medium', 'high']")

      # Get metadata and filter a bit
      for ticker in tickers:
          mc = get_metadata(ticker, 'marketCap')
          b = get_metadata(ticker, 'beta')
          pe = get_metadata(ticker, 'forwardPE')

          if (mc is not None) and (b is not None) and (pe is not None) and (pe_min <= pe <= pe_max):
              market_cap[ticker] = mc
              beta[ticker] = b
              pe_ratio[ticker] = pe
          else:
              excluded_tickers.append(ticker)

      # Print excluded tickers
      if excluded_tickers:
          print(f"Excluded tickers due to missing data: {excluded_tickers}")

      return market_cap, beta, pe_ratio

  # Retrieve metadata for all tickers
  market_cap, beta, pe_ratio = retrieve_and_filter_metadata(tickers, risk_level_stocks)

  # refine the tickers
  tickers = market_cap.keys()

  # for visualizations - restructure a bit
  market_caps = [market_cap[ticker] for ticker in tickers]
  betas = [beta[ticker] for ticker in tickers]
  pe_ratios = [pe_ratio[ticker] for ticker in tickers]
  market_caps_normalized = [mc / 1e9 for mc in market_caps]  # Convert to billions for scaling

  #

elif investment_type == "ETF":
  # Function to retrieve ETF metadata
  def retrieve_etf_metadata(tickers):
      net_assets = {}
      excluded_tickers = []

      for ticker in tickers:
          etf = yf.Ticker(ticker)
          info = etf.info

          net_assets_value = info.get('totalAssets')

          if net_assets_value is not None:
              net_assets[ticker] = net_assets_value
          else:
              excluded_tickers.append(ticker)

      # Print excluded tickers
      if excluded_tickers:
          print(f"Excluded tickers due to missing data: {excluded_tickers}")

      return net_assets

  # Retrieve metadata for all tickers
  net_assets = retrieve_etf_metadata(tickers)




# --------------------------
# --------------------------
# --------------------------
# --------------------------
# --------------------------
# --------------------------
# --------------------------


if investment_type == 'stocks':
  def filter_stocks(
      risk_level_stocks: str,
      size_preference: str,
      pe_preference: str,
      meta_marketcap: dict,
      meta_beta: dict,
      meta_pe: dict,
      marketcap_boundaries: list = [2e9, 10e9],
      pe_boundary: float = 20,
  ) -> list:
      """
      Filters stocks based on risk level and market capitalization size preference.

      Parameters:
      - risk_level_stocks (str):      Desired risk level ('low', 'medium', 'high').
      - size_preference (str):        Desired market cap size ('small', 'medium', 'large').
      - pe_preference (str):          Desired P/E ratio preference ('value', 'growth').
      - meta_marketcap (dict):        Dictionary of market capitalizations.
      - meta_beta (dict):             Dictionary of beta values.
      - meta_pe (dict):               Dictionary of P/E ratios.
      - marketcap_boundaries (list):  Boundaries for market cap classification [small_cap_max, mid_cap_max].
      - pe_boundary (float):          Boundary of value stocks `>` or growth stocks '<='

      Returns:
      - list: Filtered list of stock tickers.
      """
      filtered_tickers = []

      for ticker in meta_marketcap.keys():
          mc = meta_marketcap.get(ticker)
          b = meta_beta.get(ticker)
          pe = meta_pe.get(ticker)

          if mc is None or b is None or pe is None:
              continue

          # Market cap classification
          if size_preference == 'small' and mc > marketcap_boundaries[0]:
              continue
          if size_preference == 'medium' and (mc <= marketcap_boundaries[0] or mc > marketcap_boundaries[1]):
              continue
          if size_preference == 'large' and mc <= marketcap_boundaries[1]:
              continue

          # Risk classification based on beta
          if risk_level_stocks == 'low' and b > 1:
              continue
          if risk_level_stocks == 'medium' and (b <= 1 or b > 1.5):
              continue
          if risk_level_stocks == 'high' and b <= 1.5:
              continue

          # P/E ratio classification
          if pe_preference == 'value' and pe > pe_boundary:
              continue
          if pe_preference == 'growth' and pe <= pe_boundary:
              continue

          filtered_tickers.append(ticker)

      return filtered_tickers

  selected_tickers = filter_stocks(risk_level_stocks, size_preference, pe_preference, market_cap, beta, pe_ratio,)
  print(f"Stocks to invest in with your current profile: {selected_tickers}")

if investment_type == 'stocks':
  from typing import Union
  import requests
  # Function to get sentiment from Alpha Vantage
  def get_sentiment(
      symbol: dict,
      api_key: str = "4HJ8ICSQ87YQ91QK",
      ) -> Union[pd.DataFrame, None]:
      url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}"
      response = requests.get(url)
      data = response.json()

      # Extract sentiment data
      if "feed" in data:
          sentiment_scores = []
          for article in data['feed']:
              sentiment = article.get('overall_sentiment_score', 0)
              sentiment_scores.append(sentiment)

          # Calculate overall sentiment score
          overall_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
          return overall_sentiment
      else:
          return None


  # List of stock, ETF, and crypto symbols
  symbols = {
      'stocks': selected_tickers,
  }

  # Dataframe to store results
  sentiment_data = pd.DataFrame(columns=['Symbol','Sentiment'])

  # Loop through symbols and fetch sentiment data
  for category, tickers in symbols.items():
      for ticker in tickers:
          sentiment = get_sentiment(ticker)
          sentiment_data = pd.concat([sentiment_data, pd.DataFrame({'Symbol': [ticker], 'Sentiment': [sentiment]})], ignore_index=True)

  # Calculate overall sentiment for each category
  overall_sentiment = sentiment_data['Sentiment'].mean()

  # Display the sentiment data
  sentiment_data = sentiment_data.sort_values(by='Sentiment', ascending=False)
  overall_sentiment, sentiment_data

if investment_type == 'ETF':
  # Function to calculate standard deviation
  def calculate_standard_deviation(ticker, period=f'{time_window}y'):
      etf = yf.Ticker(ticker)
      hist = etf.history(period=period)
      returns = hist['Close'].pct_change().dropna()
      std_dev = np.std(returns)
      return std_dev

  # Function to calculate tracking error
  def calculate_tracking_error(etf_ticker, benchmark_ticker, period=f'{time_window}y'):
      etf = yf.Ticker(etf_ticker)
      benchmark = yf.Ticker(benchmark_ticker)

      etf_hist = etf.history(period=period)
      benchmark_hist = benchmark.history(period=period)

      etf_returns = etf_hist['Close'].pct_change().dropna()
      benchmark_returns = benchmark_hist['Close'].pct_change().dropna()

      # Align the data by date
      combined = etf_returns.to_frame('etf').join(benchmark_returns.to_frame('benchmark'), how='inner')

      tracking_error = np.std(combined['etf'] - combined['benchmark'])
      return tracking_error


  # Function to classify ETFs based on risk, standard deviation, and tracking error
  def classify_etfs(
        risk_level_etf: str,
        meta_net_assets: dict,
        meta_standard_deviation: dict,
        meta_tracking_error: dict,
    ) -> list:
        """
        Filters ETFs based on risk level using available data.

        Parameters:
        - risk_level_etf (str):             Desired risk level ('low', 'medium', 'high').
        - meta_net_assets (dict):           Dictionary of net assets.
        - meta_standard_deviation (dict):   Dictionary of standard deviations.
        - meta_tracking_error (dict):       Dictionary of tracking errors.

        Returns:
        - list: Filtered list of ETF tickers.
        """
        # Determine boundaries using binning
        net_assets_values = list(meta_net_assets.values())
        std_dev_values = list(meta_standard_deviation.values())
        tracking_error_values = list(meta_tracking_error.values())

        net_assets_boundary_low, net_assets_boundary_medium = np.percentile(net_assets_values, [33.33, 66.67])
        std_dev_boundary_low, std_dev_boundary_medium = np.percentile(std_dev_values, [33.33, 66.67])
        tracking_error_boundary_low, tracking_error_boundary_medium = np.percentile(tracking_error_values, [33.33, 66.67])

        filtered_tickers = []
        excluded_tickers = []

        for ticker in meta_net_assets.keys():
            na = meta_net_assets.get(ticker)
            sd = meta_standard_deviation.get(ticker)
            te = meta_tracking_error.get(ticker)

            if na is None or sd is None or te is None:
                continue

            # Low Risk:       High net assets & low or medium standard deviation & low or medium tracking error.
            # Medium Risk:    Moderate net assets & standard deviation & tracking error.
            # High Risk:      Lower net assets & medium or higher standard deviation & medium or higher tracking error.

            # Risk classification based on net assets, standard deviation, and tracking error
            if (risk_level_etf == 'low'
                  and na >= net_assets_boundary_medium
                      and sd <= std_dev_boundary_medium
                          and te <= tracking_error_boundary_medium):
              filtered_tickers.append(ticker)

            elif (risk_level_etf == 'medium'
                  and ((na < net_assets_boundary_medium and na >= net_assets_boundary_low)
                      and ((sd > std_dev_boundary_low and sd <= std_dev_boundary_medium)
                          and (te > tracking_error_boundary_low and te <= tracking_error_boundary_medium)))):
                filtered_tickers.append(ticker)

            elif (risk_level_etf == 'high'
                  and na < net_assets_boundary_low
                      and sd > std_dev_boundary_low
                          and te > tracking_error_boundary_low):
                filtered_tickers.append(ticker)

            else:
              excluded_tickers.append(ticker)

        return filtered_tickers, excluded_tickers


  # Calculate additional metrics
  standard_deviations = {ticker: calculate_standard_deviation(ticker) for ticker in tickers}
  # Using IVV (iShares Core S&P 500 ETF) as a benchmark for calculating tracking error is common because it represents a broad market index,
  tracking_errors = {ticker: calculate_tracking_error(ticker, 'IVV') for ticker in tickers}

  # Select tickers based on preferences
  selected_tickers, excluded_tickers = classify_etfs(
      risk_level_etf,
      net_assets,
      standard_deviations,
      tracking_errors,
  )

  print(f"ETFs to invest in with your current profile `{risk_level_etf}`: {selected_tickers}")


  # Calculate daily returns
returns = data[selected_tickers].pct_change(fill_method=None).dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Function to optimize portfolio for a given group of ETFs
def optimize_portfolio(
    tickers: list,
    mean_returns: np.matrix,
    cov_matrix: np.matrix,
    optimization_type: str = "minimize portfolio risk",
    risk_free_rate=0.0,
):
    """
    Optimize a portfolio for a given group of ETFs.

    Parameters:
    tickers (list): List of ticker symbols for the ETFs.
    mean_returns (np.matrix): Matrix of mean returns for the ETFs.
    cov_matrix (np.matrix): Covariance matrix of the returns for the ETFs.
    optimization_type (str): Type of optimization to perform. Options are:
        - "minimize portfolio risk": Minimize the portfolio risk (default).
        - "maximize portfolio return": Maximize the portfolio return.
        - "Risk-Adjusted Return": Maximize the Sharpe ratio.
    risk_free_rate (float): Risk-free rate used in the Sharpe ratio calculation (default is 0.0).

    Returns:
    np.ndarray: Optimized weights for the portfolio.

    Raises:
    ValueError: If an invalid optimization type is provided.

    Example:
    >>> tickers = ['AAPL', 'MSFT', 'GOOGL']
    >>> mean_returns = np.matrix([0.12, 0.10, 0.15])
    >>> cov_matrix = np.matrix([[0.005, -0.010, 0.004],
                                [-0.010, 0.040, -0.002],
                                [0.004, -0.002, 0.023]])
    >>> optimize_portfolio(tickers, mean_returns, cov_matrix, optimization_type="maximize portfolio return")
    array([0.3, 0.4, 0.3])
    """
    num_assets = len(tickers)
    weights = Variable(num_assets)
    portfolio_return = mean_returns[tickers].values @ weights
    portfolio_risk = quad_form(weights, cov_matrix.loc[tickers, tickers])
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_risk

    # Define the objective function based on optimization type
    if optimization_type == "minimize portfolio risk":
        objective = Minimize(portfolio_risk)
    elif optimization_type == "maximize portfolio return":
        objective = Maximize(portfolio_return)
    elif optimization_type == "Risk-Adjusted Return":
        objective = Maximize(sharpe_ratio)
    else:
        raise ValueError("Invalid optimization type. Choose from 'minimize portfolio risk', 'maximize portfolio return', or 'Risk-Adjusted Return'.")

    # Define the constraints (weights sum to 1)
    constraints = [weights >= 0, sum(weights) == 1]

    # Solve the optimization problem
    problem = Problem(objective, constraints)
    problem.solve()

    return weights.value
# Example: Optimize portfolio for selected ETFs
optimized_weights = optimize_portfolio(selected_tickers, mean_returns, cov_matrix, optimization_type)

# Display the optimized portfolio
portfolio = pd.DataFrame({'selected_tickers': selected_tickers, 'Weight': optimized_weights})

# # Adjust for minimal portfolio allocation and clear information
portfolio['Percentage'] = portfolio['Weight'] * 100
portfolio = portfolio[portfolio['Percentage'].abs() >= percentage_threshold]
portfolio['Percentage'] = portfolio['Percentage'].round(1)

# Ensure the total is exactly 100% by adjusting the largest weight
weight_diff = 100 - portfolio['Percentage'].sum()
max_weight_index = portfolio['Percentage'].idxmax()
portfolio.at[max_weight_index, 'Percentage'] += weight_diff

# construct weight back to adjusted fractions
portfolio['Weight'] = portfolio['Percentage'] / 100

# Calculate the EURO amount for each ticker and adjust for the fee
total_investment = buy_in_EUR - fee
portfolio['EURO Amount'] = (portfolio['Weight'] / 100) * total_investment

# if stocks we can add sentiment:
if investment_type == 'stocks':
  portfolio = pd.merge(portfolio, sentiment_data, left_on="selected_tickers", right_on="Symbol")
  portfolio = portfolio.drop(columns=["Symbol"])
  print(f"\nOverall market sentiment:{round(overall_sentiment,3)} \n")

# Show portfolio
print("pre-selected stocks before optimization:", selected_tickers)
print(f"\nOptimized portfolio: \n",portfolio)


# i.      Construct weighted signal based on fractions
# get right naming
if investment_type == 'ETF':
  risk_lvl = risk_level_etf
elif investment_type == 'stocks':
  risk_lvl = risk_level_stocks
else:
  print(f"Warning :)")

# Calculate weighted signal
weighted_signal = pd.Series(0, index=data.index)
for ticker, weight in zip(portfolio['selected_tickers'], portfolio['Weight']):
    weighted_signal += data[ticker] * weight

# Normalize the data
normalized_data = data[selected_tickers] / data[selected_tickers].iloc[0, :]

# Calculate yearly returns for weighted signal
weighted_signal_yearly = weighted_signal.resample('YE').last()
yearly_returns_weighted = weighted_signal_yearly.pct_change().dropna() * 100

# Calculate yearly returns for each ticker
yearly_returns_tickers = data[selected_tickers].resample('YE').last().pct_change().dropna() * 100

# Plot the normalized weighted signal and individual tickers
plt.figure(figsize=(16, 9))
plt.plot(weighted_signal / weighted_signal.iloc[0], label='Weighted Signal', linewidth=6)
for ticker in selected_tickers:
    plt.plot(normalized_data[ticker], label=ticker, linestyle='--', linewidth=1)
plt.title(f'Normalized Weighted Signal and Individual Selected Tickers Over Time, investment type: {investment_type}, risk profile `{risk_lvl}` & optimization type: `{optimization_type}`')
plt.xlabel('Date')
plt.ylabel('Normalized Signal Value')
plt.legend()
plt.grid(True)
plt.show()



# Construct a dataframe for yearly returns
yearly_returns_df = pd.DataFrame(yearly_returns_weighted, columns=['Weighted Signal'])
for ticker in yearly_returns_tickers:
    yearly_returns_df[ticker] = yearly_returns_tickers[ticker]

# Plot the yearly returns as a bar graph per year for each thing next to each other
# Plot the yearly returns as a bar graph per year for each thing next to each other
ax = yearly_returns_df.plot(kind='bar', figsize=(16, 9), alpha=0.7)
bars = ax.patches

ax.set_title(f'Yearly Returns for Weighted Signal and Individual selected Tickers, investment type: {investment_type} with risk profile `{risk_lvl}`')
ax.set_xlabel('Year')
ax.set_ylabel('Return (%)')
ax.legend(title='Tickers', loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(True)
ax.set_xticklabels([date.year for date in yearly_returns_df.index], rotation=0)