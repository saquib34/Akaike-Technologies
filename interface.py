import gradio as gr
import requests
import json
import os

# Get Hugging Face Space URL automatically
SPACE_URL = os.getenv("SPACE_URL", "http://localhost:7860")
API_URL = f"{SPACE_URL}/api/analyze"

def analyze_company(company_name):
    """Send request to FastAPI backend and format results"""
    try:
        response = requests.post(
            API_URL,
            json={"name": company_name},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        
        # Format results
        with gr.Blocks() as demo:
            with gr.Tabs():
                # Results Tab
                with gr.Tab("Analysis Results"):
                    gr.Markdown(f"## {data['Final Sentiment Analysis']}")
                    
                    # Sentiment Visualization
                    with gr.Row():
                        gr.BarPlot(
                            value=list(data['ComparativeSentimentScore'].values()),
                            labels=list(data['ComparativeSentimentScore'].keys()),
                            title="Sentiment Distribution",
                            color=["#4CAF50", "#F44336", "#9E9E9E"]
                        )
                        
                        # Audio Output
                        gr.Audio(
                            value=f"data:audio/wav;base64,{data['Audio']}",
                            label="Hindi Summary Audio",
                            visible=bool(data['Audio'])
                        )

                # Articles Tab
                with gr.Tab("News Articles"):
                    for idx, article in enumerate(data['Articles'], 1):
                        with gr.Accordion(f"Article {idx}: {article['Title']}", open=False):
                            gr.Markdown(f"**Summary**: {article['Summary']}")
                            gr.Markdown(f"**Sentiment**: {article['Sentiment']}")
                            gr.Markdown(f"**URL**: [Link]({article['URL']})")
                            gr.Markdown(f"**Key Topics**: {', '.join(article['Topics'])}")

                # Comparative Analysis Tab
                with gr.Tab("Comparative Analysis"):
                    with gr.Accordion("Topic Overlap", open=True):
                        gr.Markdown(f"**Common Topics**: {', '.join(data['Topic Overlap']['Common Topics'])}")
                        gr.Dataframe(
                            headers=["Article", "Unique Topics"],
                            value=[[f"Article {i+1}", ", ".join(topics)] 
                                  for i, topics in enumerate(data['Topic Overlap']['Unique Topics'])]
                        )
                    
                    with gr.Accordion("Coverage Differences", open=False):
                        gr.Dataframe(
                            headers=["Comparison", "Unique Topics"],
                            value=[[item['Comparison'], ", ".join(item['Unique Topics'])] 
                                  for item in data['Coverage Differences']]
                        )

            return demo
        
    except Exception as e:
        return gr.Markdown(f"## Error: {str(e)}")

# Create Gradio interface
with gr.Blocks(title="News Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📰 Company News Analyzer")
    with gr.Row():
        company_input = gr.Textbox(
            label="Enter Company Name",
            placeholder="e.g., Microsoft, Apple...",
            interactive=True
        )
        submit_btn = gr.Button("Analyze", variant="primary")
    
    output = gr.HTML()
    
    submit_btn.click(
        fn=analyze_company,
        inputs=company_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_port=7861)  # Different port from FastAPI