import streamlit as st
import os
from dotenv import load_dotenv
from Testing import DecisionNode

# Load environment variables from .env
load_dotenv()

# Main Streamlit App
st.title("AI Agent Based Weather Forecast and Rag Application")

option = st.selectbox("Choose Input Type", ["Weather", "PDF Query"])

user_input = st.text_input("Enter your Query or City")

uploaded_file = None
if option == "PDF Query":
    uploaded_file = st.file_uploader("Upload your PDF File", type="pdf")

# Log user interaction and execute the process
if st.button("Get Response"):
    ai_chain = DecisionNode()  # Initialize DecisionNode
    if option == "Weather":
        if user_input:  # Ensure there's input for weather
            result = ai_chain.decide(user_input)  # Call weather API
            st.success(result)
        else:
            st.warning("Please enter a valid city name for the weather query.")
    elif option == "PDF Query":
        if uploaded_file:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            result = ai_chain.decide(user_input, "temp.pdf")  # Call PDF fetch
            st.success(result)
        else:
            st.warning("Please upload a PDF file for querying.")
