from pydantic import BaseModel
from typing import Optional, Dict, Any

class AskRequest(BaseModel):
    message: str

class UploadResponse(BaseModel):
    message: str
    doc_id: str

class AskResponse(BaseModel):
    question: str
    answer: str

class ComponentStatus(BaseModel):
    api: str
    ollama: str
    vector_db: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: ComponentStatus
    details: Optional[Dict[str, Any]] = None