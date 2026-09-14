"""
Main application entry point for the RAG with Ollama service.
Configures FastAPI, lifespan events, middleware, and global exception handlers.
"""

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.exceptions import RAGAppException
from app.core.logging_config import get_logger, setup_logging
from app.services.rag_service import vector_db
from config import API_HOST, API_PORT, LOG_FILE, LOG_LEVEL, UPLOAD_DIR, VECTOR_DB_PATH

# Initialize logging system
setup_logging(log_level=LOG_LEVEL, log_file=LOG_FILE)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for application startup and shutdown tasks.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting RAG with Ollama API...")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Uploads Directory: {UPLOAD_DIR}")
    logger.info(f"Vector DB Storage: {VECTOR_DB_PATH}")

    # Ensure required directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)

    vectors_count = vector_db.index.ntotal if hasattr(vector_db, "index") and vector_db.index else 0
    logger.info(f"Vector Database ready with {vectors_count} indexed vectors.")
    logger.info("Application startup complete.")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down RAG with Ollama API gracefully...")


app = FastAPI(
    title="RAG with Ollama API",
    description="Production-grade Retrieval-Augmented Generation API powered by FAISS and Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)

# 1. Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Request Timing and Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    method = request.method
    path = request.url.path

    # Process request
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000.0
        logger.info(f"{method} {path} - {response.status_code} ({duration_ms:.1f}ms)")
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000.0
        logger.error(f"{method} {path} - FAILED ({duration_ms:.1f}ms): {e}", exc_info=True)
        raise e


# 3. Global Exception Handlers
@app.exception_handler(RAGAppException)
async def rag_app_exception_handler(request: Request, exc: RAGAppException):
    logger.error(f"Application error on {request.method} {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.message,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": exc.errors(),
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.critical(
        f"Unhandled server error on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected internal server error occurred. Please try again later.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


# 4. Include Endpoints Router
app.include_router(router)


@app.get(
    "/",
    tags=["General"],
    summary="Root health check",
    description="Quick health verification endpoint.",
)
def root():
    return {"status": "RAG API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)

