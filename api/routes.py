from fastapi import APIRouter, UploadFile, File, HTTPException
from services.rag_service import ingest_text, query_rag
from utils.file_loader import extract_text
# from db.vector_db import 
from api.schemas import AskRequest
import os
import shutil

router = APIRouter()
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        text = extract_text(file_path)
    except Exception as e:
        HTTPException(status_code=400, detail=str(e))
    
    if not text.strip():
        raise HTTPException(status_code=400, detail="No readble text found in file")

    # Ingest into RAG
    ingest_text(text)

    return {
        "message": "File uploaded and indexed successfully",
        "filename": file.filename
    }

@router.post("/ask")
def ask_question(payload: AskRequest):
    answer = query_rag(payload.message)
    return {
        "question": payload.message,
        "answer": answer
    }