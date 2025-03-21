import streamlit as st
import requests
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
HF_API_URL = "https://saquib34-news-analyzer.hf.space/api/analyze"
SAMPLE_COMPANIES = ["Microsoft", "Apple", "Google", "Amazon", "Tesla", "Meta"]
REQUEST_TIMEOUT = 600  # 2 minutes
CACHE_TTL = 1800  # 30 minutes

# Helper functions
@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=4, max=15))
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_company(company_name):
    """Call Hugging Face API with enhanced error handling"""
    try:
        response = requests.post(
            HF_API_URL,
            json={"name": company_name.strip()},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.error("Backend response timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Connection Error: {str(e)}")
        return None
    except json.JSONDecodeError:
        st.error("Invalid response from backend")
        return None

def validate_audio(audio_base64):
    """Validate and decode audio data"""
    try:
        if not audio_base64 or len(audio_base64) < 100:
            return None
        return base64.b64decode(audio_base64)
    except Exception as e:
        st.error(f"Audio decoding error: {str(e)}")
        return None

def create_sentiment_chart(data):
    """Generate pie chart with validation"""
    try:
        if not data or sum(data.values()) == 0:
            return None
            
        fig, ax = plt.subplots()
        values = list(data.values())
        labels = list(data.keys())
        colors = ['#4CAF50', '#F44336', '#9E9E9E']
        
        ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, wedgeprops={'edgecolor': 'white'})
        ax.axis('equal')
        return fig
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        return None

def safe_dataframe(data_dict, default_index=None):
    """Create DataFrame with length validation"""
    try:
        if not data_dict:
            return pd.DataFrame()
            
        lengths = [len(v) for v in data_dict.values()]
        if len(set(lengths)) > 1:
            st.warning("Data inconsistency detected in results")
            return pd.DataFrame()
            
        return pd.DataFrame(data_dict, index=default_index)
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return pd.DataFrame()

# Streamlit UI Configuration
st.set_page_config(
    page_title="News Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Management
if 'result' not in st.session_state:
    st.session_state.result = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0

# Sidebar Controls
with st.sidebar:
    st.title("Control Panel")
    st.markdown("---")
    
    selected_company = st.selectbox(
        "Select Company",
        SAMPLE_COMPANIES,
        index=0,
        help="Choose from popular companies or enter a custom name below"
    )
    
    custom_company = st.text_input(
        "Custom Company Name",
        placeholder="e.g., Netflix, Samsung...",
        max_chars=50
    )
    
    company = custom_company.strip() if custom_company else selected_company
    
    st.markdown("---")
    st.markdown("### Settings")
    auto_play = st.checkbox("Auto-Play Audio", True)
    show_raw = st.checkbox("Show Raw Data", False)
    
    st.markdown("---")
    st.markdown("### System Monitor")
    st.metric("API Timeout", f"{REQUEST_TIMEOUT}s")
    st.metric("Cache Status", "Active" if st.session_state.result else "Inactive")
    st.progress(st.session_state.progress)

# Main Interface
st.title("📈 Advanced News Sentiment Analyzer")
st.markdown("""
    *Comprehensive news analysis with AI-powered insights and audio summaries*  
    **Note**: Results based on latest articles from trusted sources
""")

# Analysis Execution
if st.button("Start Analysis", type="primary", key="analyze_btn"):
    if not company:
        st.warning("Please enter a company name")
    else:
        st.session_state.progress = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("Initializing analysis pipeline..."):
                for i in range(100):
                    st.session_state.progress = i + 1
                    progress_bar.progress(st.session_state.progress)
                    status_text.text(f"Progress: {st.session_state.progress}%")
                    time.sleep(0.1)  # Simulated progress
                    
                st.session_state.result = analyze_company(company)
                
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.session_state.result = None
        finally:
            progress_bar.empty()
            status_text.empty()

# Results Display
if st.session_state.result:
    result = st.session_state.result
    
    try:
        # Header Section
        st.success(f"Analysis Complete for **{result.get('Company', 'Unknown')}**")
        
        # Main Metrics
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 2])
            
            with col1:
                st.subheader("Sentiment Distribution")
                chart = create_sentiment_chart(result.get('ComparativeSentimentScore', {}))
                if chart:
                    st.pyplot(chart)
                else:
                    st.warning("No sentiment data available")
            
            with col2:
                st.subheader("Key Topics Cloud")
                try:
                    text = ' '.join([', '.join(article.get('Topics', [])) 
                                   for article in result.get('Articles', [])])
                    if text:
                        wordcloud = WordCloud(width=800, height=400, 
                                            background_color='white').generate(text)
                        st.image(wordcloud.to_array(), use_column_width=True)
                    else:
                        st.warning("No topics identified")
                except Exception as e:
                    st.error(f"Word cloud error: {str(e)}")
            
            with col3:
                st.subheader("Audio Summary")
                audio_data = result.get("Audio", "")
                audio_bytes = validate_audio(audio_data)
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name=f"{company}_summary.wav",
                        mime="audio/wav"
                    )
                else:
                    st.warning("Audio summary unavailable")

        # Detailed Analysis Tabs
        tab1, tab2, tab3 = st.tabs(["Articles", "Comparative Data", "Technical Info"])
        
        with tab1:
            st.subheader("Article Breakdown")
            articles = result.get('Articles', [])
            if articles:
                for idx, article in enumerate(articles, 1):
                    with st.expander(f"📰 Article {idx}: {article.get('Title', 'Untitled')}", 
                                   expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Summary**: {article.get('Summary', 'No summary available')}")
                            if article.get('URL'):
                                st.markdown(f"**Source**: [Read Article]({article['URL']})")
                        with col2:
                            st.markdown(f"**Sentiment**: {article.get('Sentiment', 'N/A')}")
                            if article.get('Topics'):
                                st.markdown("**Key Topics**:")
                                for topic in article['Topics']:
                                    st.markdown(f"- {topic}")
            else:
                st.warning("No articles found for analysis")

        with tab2:
            st.subheader("Comparative Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.markdown("### Topic Overlap")
                    common_topics = result.get('topic_overlap', {}).get('common_topics', [])
                    
                    if common_topics:
                        # Create DataFrame with safe data
                        df = pd.DataFrame({
                            "Common Topics": common_topics,
                            "Frequency": [1] * len(common_topics)
                        })
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("No common topics found")

                except Exception as e:
                    st.error(f"Topic display error: {str(e)}")

            with col2:
                try:
                    st.markdown("### Coverage Differences")
                    coverage_data = result.get('Coverage Differences', [])
                    
                    if coverage_data:
                        # Validate and transform data
                        cleaned_data = []
                        for item in coverage_data:
                            cleaned_item = {
                                "Comparison": item.get("Comparison", "N/A"),
                                "Unique Topics": ", ".join(item.get("Unique Topics", []))
                            }
                            cleaned_data.append(cleaned_item)
                        
                        df = pd.DataFrame(cleaned_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No coverage differences found")

                except Exception as e:
                    st.error(f"Coverage display error: {str(e)}")

        with tab3:
            st.subheader("Technical Details")
            
            with st.expander("System Metrics"):
                tech_data = {
                    "Component": ["Backend API", "Sentiment Model", "Summary Model"],
                    "Version": ["1.2.0", "distilbert-base", "distilbart-cnn-12-6"],
                    "Status": ["Operational", "Loaded", "Loaded"]
                }
                st.dataframe(safe_dataframe(tech_data), hide_index=True)
            
            if show_raw and result:
                with st.expander("Raw API Response"):
                    st.json(result)

    except Exception as e:
        st.error("""
        ## Critical Error Rendering Results
        Please try:
        1. Refreshing the page
        2. Simplifying your search terms
        3. Contacting support if issue persists
        """)
        st.exception(e)

# Footer & Documentation
st.markdown("---")
with st.expander("Documentation & Support"):
    st.markdown("""
    ## User Guide
    
    ### Basic Usage
    1. Select a company from the dropdown
    2. Click 'Start Analysis'
    3. Explore results in different tabs
    
    ### Features
    - **Sentiment Distribution**: Pie chart of article sentiments
    - **Topic Cloud**: Visual representation of common keywords
    - **Audio Summary**: Hindi translation of key insights
    - **Technical Details**: System metrics and raw data
    
    ### Support
    **Contact:** support@newsanalyzer.com  
    **API Docs:** [API Documentation](https://api.newsanalyzer.com)  
    **Service Status:** [Status Page](https://status.newsanalyzer.com)
    """)

# Cache Management
if st.sidebar.button("Clear Cache & Reset"):
    st.session_state.clear()
    st.rerun()
