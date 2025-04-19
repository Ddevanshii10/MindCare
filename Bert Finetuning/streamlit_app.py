import streamlit as st
import requests
import json

st.title("Anxiety Detection and Sentiment Analysis")

# Text input
user_input = st.text_area("Enter your text here:", height=150)

# Create two columns for the buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Detect Anxiety"):
        if user_input:
            # Make API call to FastAPI endpoint
            response = requests.post(
                "http://localhost:8000/predict",
                json={"text": user_input}
            )
            if response.status_code == 200:
                result = response.json()
                st.write("### Anxiety Detection Result:")
                st.write(f"Predicted Label: **{result['predicted_label']}**")
            else:
                st.error("Error occurred while processing the request")
        else:
            st.warning("Please enter some text first")

with col2:
    if st.button("Analyze Sentiment"):
        if user_input:
            # Make API call to FastAPI endpoint
            response = requests.post(
                "http://localhost:8000/sentiment",
                json={"text": user_input}
            )
            if response.status_code == 200:
                result = response.json()
                st.write("### Sentiment Analysis Results:")
                scores = result['sentiment_scores']
                st.write(f"- Positive: **{scores['pos']:.3f}**")
                st.write(f"- Negative: **{scores['neg']:.3f}**")
                st.write(f"- Neutral: **{scores['neu']:.3f}**")
                st.write(f"- Compound: **{scores['compound']:.3f}**")
            else:
                st.error("Error occurred while processing the request")
        else:
            st.warning("Please enter some text first")