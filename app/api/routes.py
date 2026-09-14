"""
FastAPI router defining REST endpoints for document upload, retrieval, and health checks.
"""

from datetime import datetime
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse
import requests

from app.api.schemas import (
    AskRequest,
    AskResponse,
    BulkUploadResponse,
    FileIndexStatus,
    HealthResponse,
    UploadResponse,
)
from app.core.exceptions import (
    DocumentExtractionError,
    EmbeddingError,
    LLMGenerationError,
    RAGAppException,
)
from app.core.logging_config import get_logger
from app.services.embedding_service import get_embedding
from app.services.rag_service import ingest_document, query_rag, vector_db
from app.utils.file_loader import extract_text
from config import OLLAMA_BASE_URL, UPLOAD_DIR

logger = get_logger(__name__)
router = APIRouter()

# Ensure uploads directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and index a document",
    description="Upload a document (.pdf, .docx, .pptx, .csv, .xlsx, .json, .txt, .md) to extract text and index into FAISS.",
)
async def upload_file(file: UploadFile = File(...)):
    """
    Handle document upload, text extraction, chunking, and vector indexing.
    """
    if not file.filename or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )

    # Sanitize filename to prevent directory traversal attacks
    safe_filename = Path(file.filename).name
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    logger.info(f"Received file upload request for: '{safe_filename}'")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file '{safe_filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file on server.",
        ) from e

    try:
        text = extract_text(file_path)
    except DocumentExtractionError as e:
        logger.warning(f"Document extraction error for '{safe_filename}': {e}")
        # Clean up unparseable file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error extracting text from '{safe_filename}': {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading file content: {str(e)}",
        ) from e

    if not text or not text.strip():
        logger.warning(f"No readable text extracted from '{safe_filename}'.")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text found in document.",
        )

    try:
        chunks_count = ingest_document(safe_filename, text)
    except (EmbeddingError, RAGAppException) as e:
        logger.error(f"Failed ingesting document '{safe_filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to index document: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error ingesting '{safe_filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during indexing: {str(e)}",
        ) from e

    logger.info(f"File '{safe_filename}' successfully indexed ({chunks_count} chunks).")
    return {
        "message": "File indexed successfully",
        "doc_id": safe_filename,
        "chunks_indexed": chunks_count,
    }


@router.post(
    "/upload/bulk",
    response_model=BulkUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Bulk upload and index multiple documents",
    description="Upload multiple documents at once (.pdf, .docx, .pptx, .csv, .xlsx, .json, .txt, .md) to extract text and index into FAISS.",
)
async def upload_bulk_files(files: list[UploadFile] = File(...)):
    """
    Handle bulk document uploads. Processes each file, indexing valid documents
    and reporting per-file success/failure details without halting the entire batch.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for upload.",
        )

    logger.info(f"Received bulk upload request with {len(files)} files.")

    results: list[FileIndexStatus] = []
    successful_count = 0
    failed_count = 0
    total_chunks = 0

    for file in files:
        if not file.filename or not file.filename.strip():
            logger.warning("Skipping file with missing or empty filename in bulk upload.")
            results.append(
                FileIndexStatus(
                    filename="unknown",
                    status="failed",
                    chunks_indexed=0,
                    error="Filename is missing or empty.",
                )
            )
            failed_count += 1
            continue

        safe_filename = Path(file.filename).name
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        logger.info(f"Bulk upload processing: '{safe_filename}'")

        # 1. Save file to disk
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save file '{safe_filename}' during bulk upload: {e}", exc_info=True)
            results.append(
                FileIndexStatus(
                    filename=safe_filename,
                    status="failed",
                    chunks_indexed=0,
                    error="Failed to save file on server.",
                )
            )
            failed_count += 1
            continue

        # 2. Extract text
        try:
            text = extract_text(file_path)
        except DocumentExtractionError as e:
            logger.warning(f"Extraction error for '{safe_filename}' in bulk upload: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            results.append(
                FileIndexStatus(
                    filename=safe_filename,
                    status="failed",
                    chunks_indexed=0,
                    error=str(e),
                )
            )
            failed_count += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error extracting text from '{safe_filename}': {e}", exc_info=True)
            if os.path.exists(file_path):
                os.remove(file_path)
            results.append(
                FileIndexStatus(
                    filename=safe_filename,
                    status="failed",
                    chunks_indexed=0,
                    error=f"Error reading file content: {str(e)}",
                )
            )
            failed_count += 1
            continue

        # 3. Validate extracted text
        if not text or not text.strip():
            logger.warning(f"No readable text extracted from '{safe_filename}' in bulk upload.")
            if os.path.exists(file_path):
                os.remove(file_path)
            results.append(
                FileIndexStatus(
                    filename=safe_filename,
                    status="failed",
                    chunks_indexed=0,
                    error="No readable text found in document.",
                )
            )
            failed_count += 1
            continue

        # 4. Ingest into VectorDB
        try:
            chunks_indexed = ingest_document(safe_filename, text)
            results.append(
                FileIndexStatus(
                    filename=safe_filename,
                    status="indexed",
                    chunks_indexed=chunks_indexed,
                    error=None,
                )
            )
            successful_count += 1
            total_chunks += chunks_indexed
            logger.info(f"Bulk upload successfully indexed '{safe_filename}' ({chunks_indexed} chunks).")
        except Exception as e:
            logger.error(f"Failed ingesting '{safe_filename}' in bulk upload: {e}", exc_info=True)
            results.append(
                FileIndexStatus(
                    filename=safe_filename,
                    status="failed",
                    chunks_indexed=0,
                    error=f"Failed to index document: {str(e)}",
                )
            )
            failed_count += 1

    summary_msg = (
        f"Bulk upload complete: {successful_count} indexed, {failed_count} failed "
        f"out of {len(files)} total files ({total_chunks} total chunks)."
    )
    logger.info(summary_msg)

    return BulkUploadResponse(
        message=summary_msg,
        total_files=len(files),
        successful_uploads=successful_count,
        failed_uploads=failed_count,
        total_chunks_indexed=total_chunks,
        files=results,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Query documents (JSON response)",
    description="Ask a question about indexed documents and receive a JSON response with answer and metadata.",
)
def ask_question(payload: AskRequest):
    """
    Ask a question about uploaded documents and get a structured JSON response.
    """
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question message cannot be empty.",
        )

    try:
        rag_result = query_rag(payload.message)
    except (LLMGenerationError, EmbeddingError) as e:
        logger.error(f"Upstream service error during Q&A: {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during Q&A: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while answering your question.",
        ) from e

    return {
        "question": payload.message,
        "answer": rag_result["answer"],
        "sources_found": rag_result.get("sources_found", 0),
        "is_relevant": rag_result.get("is_relevant", False),
        "reasoning": rag_result.get("reasoning"),
    }


@router.post(
    "/ask/text",
    response_class=PlainTextResponse,
    status_code=status.HTTP_200_OK,
    summary="Query documents (Plain text response)",
    description="Ask a question and receive the answer directly as formatted plain text with preserved line breaks.",
)
def ask_question_text(payload: AskRequest):
    """
    Ask a question and receive the answer formatted as text/plain.
    """
    if not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question message cannot be empty.",
        )

    try:
        rag_result = query_rag(payload.message)
        return rag_result["answer"]
    except (LLMGenerationError, EmbeddingError) as e:
        logger.error(f"Upstream service error during Q&A (text): {e}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error during Q&A (text): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while answering your question.",
        ) from e


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System health check",
    description="Check overall status and connectivity to Ollama and Vector DB.",
)
def health_check():
    """
    Comprehensive health check verifying API, Ollama connectivity, and Vector DB status.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "running",
            "ollama": "unknown",
            "vector_db": "unknown",
        },
        "details": {},
    }

    # 1. Check Ollama connectivity
    try:
        ollama_tags_url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        response = requests.get(ollama_tags_url, timeout=5)
        if response.status_code == 200:
            health_status["components"]["ollama"] = "running"
        else:
            health_status["components"]["ollama"] = "error"
            health_status["status"] = "degraded"
            health_status["details"]["ollama_error"] = f"HTTP {response.status_code}"
    except Exception as e:
        health_status["components"]["ollama"] = "unavailable"
        health_status["status"] = "degraded"
        health_status["details"]["ollama_error"] = str(e)

    # 2. Check Vector DB status
    try:
        if hasattr(vector_db, "index") and vector_db.index is not None:
            health_status["components"]["vector_db"] = "running"
            health_status["details"]["vector_db_vectors"] = vector_db.index.ntotal
        else:
            health_status["components"]["vector_db"] = "error"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["vector_db"] = "error"
        health_status["status"] = "degraded"
        health_status["details"]["vector_db_error"] = str(e)

    # 3. Check embedding model
    try:
        test_embedding = get_embedding("health check probe")
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