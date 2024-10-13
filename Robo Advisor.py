import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

        # Minimal threshold for portfolio allocation (common for both stocks and ETFs)
        percentage_threshold = st.slider("Minimal threshold of portfolio allocation (%)", 
                                         min_value=0.0, max_value=100.0, value=1.0)

        # Submit button
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            # Store inputs in session state
            st.session_state['investment_type'] = investment_type
            st.session_state['percentage_threshold'] = percentage_threshold
            if investment_type == "Stocks":
                st.session_state['risk_level'] = risk_level
                st.session_state['size_preference'] = size_preference
                st.session_state['pe_preference'] = pe_preference
            elif investment_type == "ETFs":
                st.session_state['risk_profile'] = risk_profile

            st.session_state['submitted'] = True
            st.experimental_rerun()

    # The modal button is placed **outside** the form, so it doesn't conflict with form behavior
    if investment_type == "Stocks" and st.button("More information about Value vs. Growth stocks"):
        with st.modal("Information about Value vs. Growth Stocks"):
            st.markdown("""
            - **Value stocks**: Suitable for conservative investors looking for steady returns and lower risk. 
            These companies may be undervalued and typically offer dividends.
            
            - **Growth stocks**: Suitable for aggressive investors willing to take on more risk for the possibility of higher returns. 
            These companies are expected to grow at an above-average rate and often reinvest their earnings instead of offering dividends.
            """)

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
                "Minimal Portfolio Allocation Threshold (%)"
            ],
            "Your Answer": [
                st.session_state['investment_type'], 
                st.session_state['risk_level'], 
                st.session_state['size_preference'], 
                st.session_state['pe_preference'], 
                st.session_state['percentage_threshold']
            ]
        }
    elif st.session_state['investment_type'] == "ETFs":
        data = {
            "Question": [
                "Investment Type", 
                "Risk Profile", 
                "Minimal Portfolio Allocation Threshold (%)"
            ],
            "Your Answer": [
                st.session_state['investment_type'], 
                st.session_state['risk_profile'], 
                st.session_state['percentage_threshold']
            ]
        }

    df = pd.DataFrame(data)
    st.table(df)

    # Generate random data and plot a line graph (this could be based on actual data processing in a real app)
    x = np.arange(1, 11)  # X-axis: 10 data points
    y = np.random.rand(10) * st.session_state['percentage_threshold']  # Y-axis based on user input

    st.write("Portfolio allocation over time (simulated):")
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title("Simulated Portfolio Allocation")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Allocation (%)")
    st.pyplot(fig)

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
