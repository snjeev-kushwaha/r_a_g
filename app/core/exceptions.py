"""
Custom application exception hierarchy for the RAG system.
Inherits from both RAGAppException and standard exceptions for backward compatibility.
"""

from typing import Any, Dict, Optional


class RAGAppException(Exception):
    """Base exception for all domain-specific application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentExtractionError(RAGAppException, ValueError):
    """Raised when text cannot be extracted from a document or format is unsupported."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class EmbeddingError(RAGAppException, RuntimeError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=502, details=details)


class LLMGenerationError(RAGAppException, RuntimeError):
    """Raised when LLM response generation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=502, details=details)


class VectorDBError(RAGAppException, AssertionError, ValueError):
    """Raised when vector database operation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class DocumentNotFoundError(RAGAppException, KeyError):
    """Raised when a specified document is not found."""

    def __init__(self, doc_id: str):
        super().__init__(f"Document '{doc_id}' not found.", status_code=404)
