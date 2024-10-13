import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from cvxpy import Variable, quad_form, Problem, Minimize
from datetime import datetime

# Set page configuration to hide the menu bar and the footer
st.set_page_config(page_title="Investment Portfolio App", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")

# Hide the default Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to display the landing page (form)
def landing_page():
    st.title("Investment Portfolio Questionnaire")

    # Form for user input
    with st.form("investment_form"):
        # Question 1: Investment type
        investment_type = st.selectbox(
            "Do you want to invest in Stocks or ETFs?", 
            options=["Stocks", "ETFs"]
        )

        # Conditional questions based on whether user selects Stocks or ETFs
        if investment_type == "Stocks":
            # Question 2a: Risk level for Stocks
            risk_level = st.radio("What kind of risk level do you prefer for stocks?", 
                                  options=["Low", "Medium", "High"])

            # Question 2b: Market capitalization preference
            size_preference = st.radio("What market capitalization do you prefer?", 
                                       options=["Small", "Medium", "Large"])

            # Question 2c: Value or growth preference
            pe_preference = st.radio("Do you want to focus on Value stocks or Growth stocks?", 
                                     options=["Value", "Growth"])

        elif investment_type == "ETFs":
            # Question 3: Risk profile for ETFs
            risk_profile = st.radio("What is your risk profile for ETFs?", 
                                    options=["Conservative", "Progressive"])

        # General portfolio preferences
        st.write("Portfolio Setup")

        # Minimal threshold of portfolio allocation (default to 50%)
        percentage_threshold = st.slider("Minimal threshold of portfolio allocation (%)", 
                                         min_value=0.0, max_value=100.0, value=50.0)

        # Buy-in amount in EUR
        buy_in_EUR = st.number_input("What is your buy-in amount (EUR)?", min_value=0.0, value=100000.0)

        # Fee amount in EUR
        fee = st.number_input("What is the fee for the investment (EUR)?", min_value=0.0, value=1500.0)

        # Submit button
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            # Store inputs in session state
            st.session_state['investment_type'] = investment_type
            st.session_state['percentage_threshold'] = percentage_threshold
            st.session_state['buy_in_EUR'] = buy_in_EUR
            st.session_state['fee'] = fee
            if investment_type == "Stocks":
                st.session_state['risk_level'] = risk_level
                st.session_state['size_preference'] = size_preference
                st.session_state['pe_preference'] = pe_preference
            elif investment_type == "ETFs":
                st.session_state['risk_profile'] = risk_profile

            st.session_state['submitted'] = True
            st.rerun()

# Function to display the result page
def result_page():
    st.title("Your Investment Preferences Summary")

    # Display the user's responses in a table format
    if st.session_state['investment_type'] == "Stocks":
        data = {
            "Question": [
                "Investment Type", 
                "Risk Level", 
                "Market Capitalization Preference", 
                "Value or Growth Preference", 
                "Minimal Portfolio Allocation Threshold (%)", 
                "Buy-in Amount (EUR)", 
                "Fee (EUR)"
            ],
            "Your Answer": [
                st.session_state['investment_type'], 
                st.session_state['risk_level'], 
                st.session_state['size_preference'], 
                st.session_state['pe_preference'], 
                st.session_state['percentage_threshold'],
                st.session_state['buy_in_EUR'],
                st.session_state['fee']
            ]
        }
    elif st.session_state['investment_type'] == "ETFs":
        data = {
            "Question": [
                "Investment Type", 
                "Risk Profile", 
                "Minimal Portfolio Allocation Threshold (%)", 
                "Buy-in Amount (EUR)", 
                "Fee (EUR)"
            ],
            "Your Answer": [
                st.session_state['investment_type'], 
                st.session_state['risk_profile'], 
                st.session_state['percentage_threshold'],
                st.session_state['buy_in_EUR'],
                st.session_state['fee']
            ]
        }

    df = pd.DataFrame(data)
    st.table(df)

    # Fetching historical data from yfinance based on investment type
    if st.session_state['investment_type'] == 'Stocks':
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN',
            'TSLA', 'NVDA', 'NFLX', 'ADBE',
            'PYPL', 'INTC', 'CSCO', 'ORCL',
            'IBM', 'CRM', 'AMD', 'QCOM',
            'TXN', 'AVGO', 'MU', 'AMAT',
            'LRCX', 'KLAC', 'ASML', 'TSM',
            'BABA', 'JD', 'PDD', 'BIDU'
        ]
        
        # Function to retrieve metadata on the stocks
        def get_metadata(ticker, meta_type='marketCap'):
            stock = yf.Ticker(ticker)
            return stock.info.get(meta_type, None)

        def retrieve_and_filter_metadata(tickers):
            market_cap = {}
            beta = {}
            pe_ratio = {}
            excluded_tickers = []

            for ticker in tickers:
                mc = get_metadata(ticker, 'marketCap')
                b = get_metadata(ticker, 'beta')
                pe = get_metadata(ticker, 'forwardPE')

                if (mc is not None) and (b is not None) and (pe is not None):
                    market_cap[ticker] = mc
                    beta[ticker] = b
                    pe_ratio[ticker] = pe
                else:
                    excluded_tickers.append(ticker)

            # Print excluded tickers
            if excluded_tickers:
                st.warning(f"Excluded tickers due to missing data: {excluded_tickers}")

            return market_cap, beta, pe_ratio

        # Retrieve metadata for all tickers
        market_cap, beta, pe_ratio = retrieve_and_filter_metadata(tickers)

        # Refine the tickers
        tickers = list(market_cap.keys())  # Convert dict_keys to a list

        # For visualizations - restructure a bit
        market_caps = [market_cap[ticker] for ticker in tickers]
        betas = [beta[ticker] for ticker in tickers]
        pe_ratios = [pe_ratio[ticker] for ticker in tickers]
        market_caps_normalized = [mc / 1e9 for mc in market_caps]  # Convert to billions for scaling

        # Fetch historical price data for the selected tickers
        data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']

        # Normalize the stock prices
        normalized_data = data / data.iloc[0, :]

        # Plot the normalized prices
        st.write(f"Normalized Stock Prices from 2020 to 2023")
        plt.figure(figsize=(14, 7))
        for ticker in tickers:
            plt.plot(normalized_data[ticker], label=ticker)

        # Multi-column legend
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
        plt.title(f'Normalized Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        st.pyplot(plt)

        # Market Capitalization vs. Beta scatter plot
        plt.figure(figsize=(14, 7))
        plt.scatter(betas, market_caps_normalized, s=market_caps_normalized, alpha=0.5)
        for i, ticker in enumerate(tickers):
            plt.text(betas[i], market_caps_normalized[i], ticker, fontsize=9)
        plt.title('Market Capitalization vs. Beta')
        plt.xlabel('Beta')
        plt.ylabel('Market Capitalization (Billions)')
        plt.grid(True)
        st.pyplot(plt)

        # Market Capitalization vs. P/E Ratio scatter plot
        plt.figure(figsize=(14, 7))
        plt.scatter(pe_ratios, market_caps_normalized, s=market_caps_normalized, alpha=0.5)
        for i, ticker in enumerate(tickers):
            plt.text(pe_ratios[i], market_caps_normalized[i], ticker, fontsize=9)
        plt.title('Market Capitalization vs. P/E Ratio')
        plt.xlabel('P/E Ratio')
        plt.ylabel('Market Capitalization (Billions)')
        plt.grid(True)
        st.pyplot(plt)

        # Function to filter stocks based on user preferences
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

        # Filter stocks based on user preferences
        selected_tickers = filter_stocks(
            st.session_state['risk_level'],
            st.session_state['size_preference'],
            st.session_state['pe_preference'],
            market_cap,
            beta,
            pe_ratio,
        )

        # Display the filtered stocks
        st.write(f"Stocks to invest in based on your profile: {selected_tickers}")

    elif st.session_state['investment_type'] == 'ETFs':
        # ETF handling can be done here, similar to stocks if needed.
        st.write("ETFs section coming soon!")

# Main entry point
if 'submitted' not in st.session_state:
    landing_page()
else:
    result_page()
