from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
from keybert import KeyBERT
from deep_translator import GoogleTranslator
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
from pathlib import Path
import tempfile
# Initialize models
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

summary_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
)

kw_model = KeyBERT()

def extract_articles(company: str) -> List[Dict]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    url = f"https://www.indiatvnews.com/topic/{company}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    articles = []
    li_elements = soup.select('li.eventTracking')[:5]

    for i in li_elements:
        link = i.find('a', class_='thumb')
        request_internal = requests.get(link['href'])
        soup_internal = BeautifulSoup(request_internal.content, "html.parser")
        p_elements = soup_internal.select('p:not(:has(*))')
        merged_text = ' '.join(p.get_text(strip=True) for p in p_elements)

        summary_text = summary_pipeline(merged_text, max_length=50, min_length=20, do_sample=False)
        summary = summary_text[0]['summary_text']
        topics = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words='english')
        topic_list = [topic[0] for topic in topics]

        articles.append({
            "Title": i['title'],
            "Summary": summary,
            "URL": link['href'],
            "Topics": topic_list
        })
    return articles

def analyze_sentiment(text: str) -> Dict:
    if not text or text == 'No Summary Available':
        return {"label": "NEUTRAL", "score": 0.5}
    result = sentiment_pipeline(text)[0]
    return result

def perform_comparative_analysis(articles):
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    common_topics = set(articles[0]['Topics'])
    unique_topics = [set(article['Topics']) for article in articles]

    coverage_differences = []
    for i in range(len(articles) - 1):
        for j in range(i + 1, len(articles)):
            diff = set(articles[i]['Topics']) ^ set(articles[j]['Topics'])
            coverage_differences.append({
                "Comparison": f"{articles[i]['Title']} vs {articles[j]['Title']}",
                "Unique Topics": list(diff)
            })
            common_topics &= set(articles[j]['Topics'])

    topic_overlap = {
        "Common Topics": list(common_topics),
        "Unique Topics": [list(topics - common_topics) for topics in unique_topics]
    }
    
    for article in articles:
        sentiment_result = analyze_sentiment(article['Summary'])
        article['Sentiment'] = sentiment_result['label']
        sentiment_counts[sentiment_result['label']] += 1
    
    return sentiment_counts, coverage_differences, topic_overlap

# Add validation in your backend's generate_hindi_tts function
def generate_hindi_tts(text: str) -> str:
    temp_dir = tempfile.mkdtemp()
    try:
        # Validate input
        if not text or len(text.strip()) < 50:
            return ""
            
        # Translation
        translated = GoogleTranslator(source='en', target='hi').translate(text[:1000])
        print(f"Translated text: {translated}")
        if not translated:
            return ""
            
        # Create temp file
        temp_file = Path(temp_dir) / "hindi_summary.wav"
        
        # Generate audio
        tts = gTTS(translated, lang='hi')
        tts.save(temp_file)
        
        # Read and encode
        with open(temp_file, "rb") as f:
            audio_bytes = f.read()
            
        return base64.b64encode(audio_bytes).decode('utf-8')
        
    except Exception as e:
        print(f"Audio Generation Error: {str(e)}")
        return ""
    finally:
        # Cleanup - remove temp directory and contents
        try:
            for file in Path(temp_dir).glob("*"):
                file.unlink()
            Path(temp_dir).rmdir()
        except Exception as cleanup_error:
            print(f"Cleanup error: {str(cleanup_error)}")
