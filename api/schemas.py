from pydantic import BaseModel

class AskRequest(BaseModel):
    message: str