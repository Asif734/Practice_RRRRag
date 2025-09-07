from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    session_id: Optional[str] = "default"
   

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
