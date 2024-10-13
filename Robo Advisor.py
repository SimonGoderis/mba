import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

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
            st.experimental_rerun()

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
    elif st.session_state['investment_type'] == 'ETFs':
        tickers = [
            'SPY', 'IVV', 'VOO', 'QQQ', 'VTI',
            'IWM', 'VUG', 'VTV', 'EEM', 'EFA'
        ]

    # Fetch historical price data for the selected tickers
    data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Adj Close']

    # Normalize the stock prices
    normalized_data = data / data.iloc[0, :]

    # Plot the normalized prices
    st.write(f"Normalized {st.session_state['investment_type']} Prices from 2020 to 2023")
    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(normalized_data[ticker], label=ticker)

    # Multi-column legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2)
    plt.title(f'Normalized {st.session_state["investment_type"]} Prices')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    st.pyplot(plt)

    # Button to go back to the Landing Page
    if st.button("Go back to Questionnaire"):
        st.session_state['submitted'] = False
        st.experimental_rerun()

# Main function to handle page rendering
def main():
    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False

    # Check if form is submitted, if not show the form page
    if not st.session_state['submitted']:
        landing_page()
    else:
        result_page()

if __name__ == "__main__":
    main()
