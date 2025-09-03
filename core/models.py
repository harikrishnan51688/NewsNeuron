from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    TIMELINE = "timeline"
    RELATIONSHIP = "relationship"

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
    
    
    

# Add these flash card models to your existing models.py
class Flashcard(BaseModel):
    id: str
    category: str
    question: str
    answer: str
    difficulty: str = "medium"  # easy, medium, hard
    tags: List[str] = []
    status: str = "new"  # new, learning, known
    created_at: datetime = datetime.now()
    last_reviewed: Optional[datetime] = None
    review_count: int = 0

class FlashcardRequest(BaseModel):
    topic: str
    difficulty: Optional[str] = "medium"
    count: int = 5
    source_type: str = "knowledge_base"  # knowledge_base, recent_news, entities

class StudySession(BaseModel):
    session_id: str
    cards_studied: int
    correct_answers: int
    duration_minutes: int
    topics_covered: List[str]
