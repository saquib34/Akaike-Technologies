import streamlit as st
import requests

# Hugging Face Space API Endpoint
HF_API_URL = "https://saquib34-news-analyzer.hf.space/api/analyze"

def analyze_company(company_name):
    """Call Hugging Face API endpoint"""
    try:
        response = requests.post(
            HF_API_URL,
            json={"name": company_name},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

# Streamlit UI
st.title("📰 Company News Analyzer")

# Input Section
company = st.text_input("Enter Company Name", placeholder="e.g., Microsoft, Apple...")

if st.button("Analyze"):
    if not company:
        st.warning("Please enter a company name")
    else:
        with st.spinner("Analyzing news coverage..."):
            result = analyze_company(company)
        
        if result:
            # Display Results
            st.subheader(result["Final Sentiment Analysis"])
            
            # Sentiment Summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Positive Articles", result["ComparativeSentimentScore"]["POSITIVE"])
                st.metric("Negative Articles", result["ComparativeSentimentScore"]["NEGATIVE"])
            with col2:
                st.metric("Neutral Articles", result["ComparativeSentimentScore"]["NEUTRAL"])
            
            # Articles Display
            st.subheader("News Articles Analysis")
            for idx, article in enumerate(result["Articles"], 1):
                with st.expander(f"Article {idx}: {article['Title']}"):
                    st.write(f"**Summary**: {article['Summary']}")
                    st.write(f"**Sentiment**: {article['Sentiment']}")
                    st.write(f"**URL**: [Read Full Article]({article['URL']})")
                    st.write(f"**Key Topics**: {', '.join(article['Topics'])}")
            
            # Audio Section
            if result["Audio"]:
                st.subheader("Hindi Summary Audio")
                st.audio(result["Audio"], format="audio/wav")
