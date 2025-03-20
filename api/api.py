from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from utils.utils import (
    extract_articles,
    perform_comparative_analysis,
    generate_hindi_tts
)

router = APIRouter()

class Company(BaseModel):
    name: str

@router.post("/api/analyze")
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

    return {
        "Company": company_name,
        "Articles": articles,
        "ComparativeSentimentScore": sentiment_counts,
        "Coverage Differences": coverage_differences,
        "Topic Overlap": topic_overlap,
        "Final Sentiment Analysis": final_analysis,
        "Audio": audio_path
    }