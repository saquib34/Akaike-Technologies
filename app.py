from fastapi import FastAPI
from api.api import router as api_router
from interface import demo as gradio_app

app = FastAPI(root_path="/")

# Mount API routes
app.include_router(api_router)

# Mount Gradio interface at root
app.mount("/", gradio_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)