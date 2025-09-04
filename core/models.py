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

# ============================================================================
# FLASHCARD MODELS
# ============================================================================

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class FlashcardStatus(str, Enum):
    NEW = "new"
    LEARNING = "learning"
    KNOWN = "known"
    REVIEW = "review"

class SourceType(str, Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    RECENT_NEWS = "recent_news" 
    ENTITIES = "entities"
    NEWS = "news"

class Flashcard(BaseModel):
    id: str
    category: str
    question: str
    answer: str
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    tags: List[str] = []
    status: FlashcardStatus = FlashcardStatus.NEW
    created_at: datetime = Field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = None
    review_count: int = 0
    success_rate: float = 0.0
    next_review: Optional[datetime] = None
    source_type: Optional[SourceType] = None
    news_date: Optional[datetime] = None
    source_url: Optional[str] = None
    context: Optional[str] = None

class FlashcardRequest(BaseModel):
    topic: Optional[str] = None
    source_type: SourceType = SourceType.NEWS
    count: int = Field(default=10, ge=1, le=50)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM

class FlashcardResponse(BaseModel):
    success: bool
    flashcards: List[Flashcard] = []
    count: int = 0
    message: str = ""
    topic: Optional[str] = None
    source_type: Optional[SourceType] = None
    error: Optional[str] = None

class ReviewRequest(BaseModel):
    correct: bool
    difficulty_rating: Optional[int] = Field(None, ge=1, le=5)
    response_time_ms: Optional[int] = None

class ReviewResponse(BaseModel):
    success: bool
    message: str
    next_review: Optional[datetime] = None
    new_status: Optional[FlashcardStatus] = None
    success_rate: Optional[float] = None
    error: Optional[str] = None

class StudySession(BaseModel):
    session_id: str
    cards: List[Flashcard] = []
    total_cards: int = 0
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    topics: List[str] = []
    cards_studied: int = 0
    correct_answers: int = 0
    duration_minutes: Optional[int] = None

class StudySessionRequest(BaseModel):
    topics: Optional[List[str]] = None
    max_cards: int = Field(default=20, ge=1, le=100)
    difficulty: Optional[DifficultyLevel] = None
    include_due: bool = True
    include_new: bool = True

class StudyStatistics(BaseModel):
    total_cards: int = 0
    cards_due: int = 0
    cards_new: int = 0
    cards_learning: int = 0
    cards_known: int = 0
    overall_success_rate: float = 0.0
    cards_studied_today: int = 0
    streak_days: int = 0
    categories: Dict[str, int] = {}
    recent_activity: List[Dict[str, Any]] = []

class FlashcardFilters(BaseModel):
    category: Optional[str] = None
    difficulty: Optional[DifficultyLevel] = None
    status: Optional[FlashcardStatus] = None
    due_for_review: bool = False
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    tags: Optional[List[str]] = None
    source_type: Optional[SourceType] = None