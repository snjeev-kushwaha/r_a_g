"""
Pydantic schemas and models for API request and response validation.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Schema for document query request."""
    message: str = Field(
        ...,
        min_length=1,
        description="The question or prompt to ask about uploaded documents.",
        examples=["What are the eligibility requirements for remote work?"]
    )


class UploadResponse(BaseModel):
    """Schema for single file upload response."""
    message: str
    doc_id: str
    chunks_indexed: Optional[int] = None


class FileIndexStatus(BaseModel):
    """Status details for an individual file in a bulk upload."""
    filename: str
    status: str  # "indexed" or "failed"
    chunks_indexed: int = 0
    error: Optional[str] = None


class BulkUploadResponse(BaseModel):
    """Schema for bulk file upload response."""
    message: str
    total_files: int
    successful_uploads: int
    failed_uploads: int
    total_chunks_indexed: int
    files: list[FileIndexStatus]


class AskResponse(BaseModel):
    """Schema for document query response."""
    question: str
    answer: str
    sources_found: Optional[int] = None
    is_relevant: Optional[bool] = None
    reasoning: Optional[str] = None


class ComponentStatus(BaseModel):
    """Component-level health check statuses."""
    api: str
    ollama: str
    vector_db: str


class HealthResponse(BaseModel):
    """Schema for comprehensive system health check."""
    status: str
    timestamp: str
    components: ComponentStatus
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response schema."""
    error: str
    detail: Optional[str] = None
    status_code: int