import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration to hide the menu bar and the footer
st.set_page_config(page_title="Two Page App", page_icon="📊", layout="centered", initial_sidebar_state="collapsed")

# Hide the default Streamlit menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Page navigation
def main():
    # Page navigation using radio buttons (placed at the top instead of sidebar)
    page = st.radio("Navigation", ["Landing Page", "Result Page"], index=0)

    # Store inputs in session state
    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False

    if page == "Landing Page":
        landing_page()
    elif page == "Result Page" and st.session_state['submitted']:
        result_page()
    else:
        st.warning("Please complete the form on the Landing Page first.")

# Landing page with a form
def landing_page():
    st.title("Landing Page")

    with st.form("user_input_form"):
        st.write("Please answer the following questions:")

        # Yes/No question
        yes_no = st.radio("Do you agree with the terms?", ("Yes", "No"))

        # Quantitative input (Slider)
        number_input = st.slider("On a scale from 1 to 100, how do you rate our service?", 1, 100, 50)

        # Submit button
        submit = st.form_submit_button("Submit")

        if submit:
            # Store data in session state
            st.session_state['yes_no'] = yes_no
            st.session_state['number_input'] = number_input
            st.session_state['submitted'] = True

            st.success("Form submitted! You can now view the results on the Result Page.")

# Result page with a table and line graph
def result_page():
    st.title("Result Page")

    # Display the user's answers in a table format
    st.write("Here are your responses:")

    # Creating a DataFrame to show data in tabular format
    data = {
        "Question": ["Do you agree with the terms?", "Service rating (1-100)"],
        "Your Answer": [st.session_state['yes_no'], st.session_state['number_input']]
    }
    df = pd.DataFrame(data)
    st.table(df)

    # Simulate some data for the line graph
    x = np.arange(1, 11)  # X-axis: 10 data points
    y = np.random.rand(10) * st.session_state['number_input']  # Y-axis based on the user input

    st.write("Line graph based on your service rating:")
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title("Sample Line Graph")
    ax.set_xlabel("X-axis (Time)")
    ax.set_ylabel("Y-axis (Values)")
    st.pyplot(fig)

    # Button to go back to the Landing Page
    if st.button("Go back to Landing Page"):
        st.session_state['submitted'] = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()
