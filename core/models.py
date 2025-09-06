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
# TIMELINE MODELS
# ============================================================================

class EventType(str, Enum):
    NEWS = "news"
    SOCIAL = "social"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    ANNOUNCEMENT = "announcement"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class TimelineEvent(BaseModel):
    id: str
    title: str
    description: str
    date: datetime
    event_type: EventType
    sentiment: SentimentType
    relevance_score: float
    entities: List[str] = []
    source_article_id: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = {}

class TimelineRequest(BaseModel):
    entity_name: str
    time_range: str = Field(default="30d", description="7d, 30d, 90d, 1y, or custom")
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=500)

class TimelineResponse(BaseModel):
    success: bool
    entity_name: str
    events: List[TimelineEvent] = []
    total_events: int = 0
    time_range: str
    start_date: datetime
    end_date: datetime
    statistics: Dict[str, Any] = {}
    related_entities: List[Dict[str, Any]] = []
    error: Optional[str] = None

class EntityMention(BaseModel):
    entity_name: str
    mention_count: int
    relevance_score: float
    first_mentioned: datetime
    last_mentioned: datetime
    sentiment_distribution: Dict[str, int] = {}

class TimelineSummary(BaseModel):
    entity_name: str
    total_events: int
    time_span_days: int
    key_developments: List[str] = []
    sentiment_trend: List[Dict[str, Any]] = []
    related_entities: List[EntityMention] = []
    trending_topics: List[str] = []