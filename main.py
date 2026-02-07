from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="RAG with Ollama")

app.include_router(router)

@app.get("/")
def health():
    return {"status": "RAG API is running"}

