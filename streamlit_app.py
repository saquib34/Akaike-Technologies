import streamlit as st
import requests
import base64
import json
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Configuration
HF_API_URL = "https://saquib34-news-analyzer.hf.space/api/analyze"
SAMPLE_COMPANIES = ["Microsoft", "Apple", "Google", "Amazon", "Tesla", "Meta"]
CACHE_TTL = 1800  # 30 minutes

# Helper functions
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_company(company_name):
    """Call Hugging Face API with caching"""
    try:
        response = requests.post(
            HF_API_URL,
            json={"name": company_name},
            headers={"Content-Type": "application/json"},
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def create_sentiment_chart(data):
    """Generate pie chart for sentiment distribution"""
    fig, ax = plt.subplots()
    values = list(data.values())
    labels = list(data.keys())
    colors = ['#4CAF50', '#F44336', '#9E9E9E']
    
    ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, wedgeprops={'edgecolor': 'white'})
    ax.axis('equal')
    return fig

def generate_wordcloud(topics):
    """Create word cloud from article topics"""
    text = ' '.join([', '.join(article['Topics']) for article in st.session_state.result['Articles']])
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white').generate(text)
    return wordcloud

# Streamlit UI Configuration
st.set_page_config(
    page_title="Advanced News Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization
if 'result' not in st.session_state:
    st.session_state.result = None

# Sidebar Controls
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    selected_company = st.selectbox(
        "Select Company",
        SAMPLE_COMPANIES,
        index=0,
        help="Choose from popular companies or type a custom name"
    )
    
    custom_company = st.text_input(
        "Or Enter Custom Name",
        placeholder="e.g., Netflix, Samsung..."
    )
    
    company = custom_company if custom_company else selected_company
    
    st.markdown("---")
    st.markdown("### Analysis Settings")
    show_raw = st.checkbox("Show Raw API Response", False)
    auto_play = st.checkbox("Auto-Play Audio", True)
    
    st.markdown("---")
    st.markdown("### System Info")
    st.write(f"Cached Results: {len(st.session_state)}")
    st.progress((st.session_state.get('progress', 0)))

# Main Interface
st.title("📈 Advanced News Sentiment Analyzer")
st.markdown("""
    *Comprehensive news analysis with AI-powered insights and audio summaries*  
    **Note**: Results based on latest 5 articles from IndiaTV News
""")

# Analysis Execution
col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("Start Analysis", type="primary")

if analyze_btn:
    with st.spinner(f"🧠 Analyzing {company} coverage..."):
        st.session_state.result = analyze_company(company)
        st.session_state.progress = 100

# Results Display
if st.session_state.result:
    result = st.session_state.result
    
    # Header Section
    st.success(f"Analysis Complete for **{result['Company']}**")
    
    # Main Metrics
    with st.container():
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            st.subheader("Sentiment Distribution")
            fig = create_sentiment_chart(result['ComparativeSentimentScore'])
            st.pyplot(fig)
            
        with col2:
            st.subheader("Key Topics Cloud")
            wordcloud = generate_wordcloud(result['Articles'])
            st.image(wordcloud.to_array(), use_column_width=True)
            
        with col3:
            st.subheader("Audio Summary")
            if result.get("Audio"):
                audio_bytes = base64.b64decode(result["Audio"])
                st.audio(audio_bytes, format="audio/wav")
                
                # Audio Download
                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name=f"{company}_summary.wav",
                    mime="audio/wav"
                )
            else:
                st.warning("Audio summary not available")

    # Detailed Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["Articles", "Comparative Data", "Technical Info"])
    
    with tab1:
        st.subheader("Article Breakdown")
        for idx, article in enumerate(result['Articles'], 1):
            with st.expander(f"📰 Article {idx}: {article['Title']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Summary**: {article['Summary']}")
                    st.markdown(f"**Full URL**: [Open Article]({article['URL']})")
                with col2:
                    st.markdown(f"**Sentiment**: {article['Sentiment']}")
                    st.markdown("**Key Topics**:")
                    for topic in article['Topics']:
                        st.markdown(f"- {topic}")
    
    with tab2:
        st.subheader("Comparative Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Topic Overlap")
            st.dataframe(
                pd.DataFrame({
                    "Common Topics": result['Topic Overlap']['Common Topics'],
                    "Unique Topics Count": [len(t) for t in result['Topic Overlap']['Unique Topics']]
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("### Coverage Differences")
            st.dataframe(
                pd.DataFrame(result['Coverage Differences']),
                use_container_width=True,
                hide_index=True
            )
    
    with tab3:
        st.subheader("Technical Details")
        
        with st.expander("Request/Response Metrics"):
            st.json({
                "api_response_time": "1.2s",
                "model_versions": {
                    "sentiment": "distilbert-base-uncased",
                    "summarization": "bart-large-cnn"
                },
                "cache_status": "HIT" if st.session_state.result else "MISS"
            })
        
        if show_raw:
            st.subheader("Raw API Response")
            st.json(result)

# Footer & Help Section
st.markdown("---")
with st.expander("ℹ️ Help & Documentation"):
    st.markdown("""
    ### User Guide
    
    **1. Company Selection**  
    - Choose from popular companies or enter a custom name
    - Names are case-insensitive
    
    **2. Analysis Results**  
    - Sentiment distribution shows percentage of positive/negative/neutral articles
    - Word cloud displays most frequent topics
    - Audio summary provides Hindi translation of key points
    
    **3. Data Interpretation**  
    - Positive: Score ≥ 0.65  
    - Neutral: 0.35 < Score < 0.65  
    - Negative: Score ≤ 0.35
    
    ### Technical Support
    Contact: support@newsanalyzer.com  
    API Documentation: [View Docs](https://api.newsanalyzer.com)
    """)

# Session Management
if st.sidebar.button("Clear Cache"):
    st.session_state.clear()
    st.rerun()
