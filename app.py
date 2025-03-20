from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from gtts import gTTS
from keybert import KeyBERT
from deep_translator import GoogleTranslator
app = FastAPI()
import base64
from io import BytesIO
# NLP Pipelines
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",

)

summary_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",

)
from sentence_transformers import SentenceTransformer



# Then initialize KeyBERT
kw_model = KeyBERT()




# Define request body
class Company(BaseModel):
    name: str

# Extract articles
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

# Analyze sentiment
def analyze_sentiment(text: str) -> Dict:
    if not text or text == 'No Summary Available':
        return {"label": "NEUTRAL", "score": 0.5}
    result = sentiment_pipeline(text)[0]
    return result

# Comparative Analysis
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
    
    # Sentiment analysis
    for article in articles:
        sentiment_result = analyze_sentiment(article['Summary'])
        article['Sentiment'] = sentiment_result['label']
        sentiment_counts[sentiment_result['label']] += 1
    
    return sentiment_counts, coverage_differences, topic_overlap

    
# Hindi TTS
def generate_hindi_tts(text: str) -> str:
    text = text[:1000]
    # translator = Translator(to_lang="hi")
    text = GoogleTranslator(source='en', target='hi').translate(text)
    tts = gTTS(text, lang='hi')
    audio_buffer = BytesIO()
    tts.save(audio_buffer)
    audio_buffer.seek(0)

    audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
    return audio_base64

# API Route
@app.post("/api/analyze")
async def analyze(company: Company):
    company_name = company.name.strip()
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name is required")

    articles = extract_articles(company_name)
    sentiment_counts, coverage_differences, topic_overlap = perform_comparative_analysis(articles)
    summary_text = " ".join([article['Summary'] for article in articles])
    audio_path = generate_hindi_tts(summary_text)

    final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    final_analysis = f"Overall sentiment towards {company_name} is {final_sentiment.lower()}."

    response = {
        "Company": company_name,
        "Articles": articles,
        "ComparativeSentimentScore": sentiment_counts,
        "Coverage Differences": coverage_differences,
        "Topic Overlap": topic_overlap,
        "Final Sentiment Analysis": final_analysis,
        "Audio": f"/static/{audio_path}"
    }
    return response

from fastapi.staticfiles import StaticFiles
# Change this line:
app.mount("/static", StaticFiles(directory="static"), name="static")



