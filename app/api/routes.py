from fastapi import APIRouter, UploadFile, File, HTTPException
# from services.rag_service import ingest_text, query_rag
from app.services.rag_service import ingest_document, query_rag
from app.utils.file_loader import extract_text
from app.api.schemas import (
    AskRequest, 
    UploadResponse, 
    AskResponse, 
    HealthResponse
)
from app.services.embedding_service import get_embedding
from app.db.vector_db import VectorDB
import os
import shutil
import requests
from datetime import datetime

router = APIRouter()
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# @router.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)

#     # Save file
#     with open(file_path, 'wb') as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     try:
#         text = extract_text(file_path)
#     except Exception as e:
#         HTTPException(status_code=400, detail=str(e))
    
#     if not text.strip():
#         raise HTTPException(status_code=400, detail="No readble text found in file")

#     # Ingest into RAG
#     ingest_text(text)

    # return {
    #     "message": "File uploaded and indexed successfully",
    #     "filename": file.filename
    # }

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_path)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found")

    # Use filename as doc_id
    ingest_document(file.filename, text)

    return {
        "message": "File indexed successfully",
        "doc_id": file.filename
    }

@router.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    answer = query_rag(payload.message)
    return {
        "question": payload.message,
        "answer": answer
    }

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Comprehensive health check endpoint
    Checks: API status, Ollama connectivity, Vector DB status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "running",
            "ollama": "unknown",
            "vector_db": "unknown"
        },
        "details": {}
    }
    
    # Check Ollama connectivity
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            health_status["components"]["ollama"] = "running"
        else:
            health_status["components"]["ollama"] = "error"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["ollama"] = "unavailable"
        health_status["status"] = "degraded"
        health_status["details"]["ollama_error"] = str(e)
    
    # Check Vector DB status
    try:
        vector_db = VectorDB()
        if hasattr(vector_db, 'index') and vector_db.index is not None:
            health_status["components"]["vector_db"] = "running"
            health_status["details"]["vector_db_vectors"] = vector_db.index.ntotal
        else:
            health_status["components"]["vector_db"] = "error"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["vector_db"] = "error"
        health_status["status"] = "degraded"
        health_status["details"]["vector_db_error"] = str(e)
    
    # Check if embedding model works
    try:
        test_embedding = get_embedding("health check")
        if test_embedding is not None and len(test_embedding) > 0:
            health_status["details"]["embedding_model"] = "working"
        else:
            health_status["details"]["embedding_model"] = "not working"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["details"]["embedding_model"] = "error"
        health_status["status"] = "degraded"
        health_status["details"]["embedding_error"] = str(e)
    
    return health_status