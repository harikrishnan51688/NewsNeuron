from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    id: str
    title: str
    content: str
    summary: str
    url: str
    source: str
    author: Optional[str] = None
    published_date: datetime
    categories: List[str] = []
    entities: List[str] = []
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None

class QueryRequest(BaseModel):
    query: str
    query_type: Optional[QueryType] = None
    filters: Optional[Dict[str, Any]] = {}
    limit: int = Field(default=10, ge=1, le=100)

class RetrievalResult(BaseModel):
    articles: List[NewsArticle]
    total_count: int
    query_analysis: Dict[str, Any]
    processing_time: float