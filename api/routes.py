"""
FastAPI routes for the AI agent application.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.agent import agent_graph
from core.config import settings
from core.models import NewsArticle
from retrieval.fetch_news import fetch_gnews
from retrieval.vector_store import VectorStore
from core.models import (
    NewsArticle, TimelineRequest, TimelineResponse, TimelineEvent, 
    EventType, SentimentType, TimelineSummary, EntityMention
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI News Agent API",
    description="AI agent for news retrieval and knowledge graph queries",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    messages: List[str]
    stream: bool = True

class ChatMessage(BaseModel):
    role: str
    content: str

class IngestResponse(BaseModel):
    success: bool
    articles_processed: int
    message: str

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI News Agent"}

@app.get("/ingest-data")
async def ingest_news_data(
    query: str,
    background_tasks: BackgroundTasks
) -> IngestResponse:
    """
    Fetch and store news articles based on a search query.
    
    Args:
        query: Search term for news articles
        background_tasks: FastAPI background tasks
        
    Returns:
        Response with ingestion results
    """
    try:
        logger.info(f"Starting news ingestion for query: {query}")
        
        # Fetch news articles
        news_data = fetch_gnews(query)
        if not news_data or 'articles' not in news_data:
            raise HTTPException(status_code=404, detail="No articles found")
        
        articles = news_data['articles']
        processed_count = 0
        
        def process_articles():
            nonlocal processed_count
            store = VectorStore()
            
            try:
                for article_data in articles:
                    try:
                        # Create NewsArticle model
                        article = NewsArticle(
                            id=article_data.get("id", ""),
                            title=article_data.get("title", ""),
                            content=article_data.get("content", ""),
                            summary=article_data.get("description", ""),
                            url=article_data.get("url", ""),
                            source=article_data.get("source", {}).get("name", ""),
                            author=article_data.get("author"),
                            published_date=datetime.fromisoformat(
                                article_data["publishedAt"].replace("Z", "+00:00")
                            ) if article_data.get("publishedAt") else datetime.now(),
                        )
                        
                        store.insert_article(article)
                        processed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Skipping article due to error: {e}")
                        continue
                        
            finally:
                store.close()
        
        # Process articles in background
        background_tasks.add_task(process_articles)
        
        return IngestResponse(
            success=True,
            articles_processed=len(articles),
            message=f"Processing {len(articles)} articles in background"
        )
        
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the AI agent using Server-Sent Events for streaming.
    
    Args:
        request: Chat request with messages and streaming preference
        
    Returns:
        StreamingResponse for real-time chat
    """
    try:
        # Prepare message history
        messages = [SystemMessage(content=settings.SYSTEM_PROMPT)]
        for msg in request.messages:
            messages.append(HumanMessage(content=msg))
        
        if request.stream:
            return StreamingResponse(
                stream_agent_response(messages),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
        else:
            # Non-streaming response
            result = await get_complete_response(messages)
            return {"response": result}
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# ============================================================================
# STREAMING FUNCTIONS
# ============================================================================

async def stream_agent_response(messages: List):
    """
    Stream the agent's response using Server-Sent Events.
    
    Args:
        messages: List of chat messages
        
    Yields:
        SSE formatted responses
    """
    try:
        # Initialize state
        state = {"messages": messages}
        
        # Stream the graph execution
        async for chunk in agent_graph.astream(state):
            for node_name, node_output in chunk.items():
                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]
                    
                    # Handle AI messages (responses)
                    if isinstance(latest_message, AIMessage) and latest_message.content:
                        # Stream content word by word for better UX
                        words = latest_message.content.split()
                        for i, word in enumerate(words):
                            if i == 0:
                                yield f"data: {json.dumps({'type': 'content', 'data': word})}\n\n"
                            else:
                                yield f"data: {json.dumps({'type': 'content', 'data': f' {word}'})}\n\n"
                            
                            # Small delay for smoother streaming
                            await asyncio.sleep(0.01)
                    
                    # Handle tool calls
                    elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'data': {'name': tool_call['name'], 'args': tool_call.get('args', {})}})}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'done', 'data': None})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

async def get_complete_response(messages: List) -> str:
    """
    Get complete response without streaming.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Complete response string
    """
    try:
        state = {"messages": messages}
        result = await agent_graph.ainvoke(state)
        
        if result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        
        return "I apologize, but I couldn't generate a response."
        
    except Exception as e:
        logger.error(f"Complete response error: {e}")
        return f"Error: {str(e)}"

# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@app.get("/search-articles")
async def search_stored_articles(query: str, limit: int = 5):
    """
    Search stored articles by semantic similarity.
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching articles
    """
    try:
        store = VectorStore()
        embedding = store.embedding_generator.generate_embeddings(query)
        results = store.pinecone_index.query(
            vector=embedding, 
            top_k=limit, 
            include_metadata=True
        )
        
        articles = []
        with store.pg_conn.cursor() as cursor:
            for match in results["matches"]:
                article_id = match["metadata"]["article_id"]
                cursor.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
                row = cursor.fetchone()
                
                if row:
                    articles.append({
                        "id": row[0],
                        "title": row[1],
                        "summary": row[3],
                        "url": row[4],
                        "source": row[5],
                        "published_date": str(row[7]),
                        "similarity_score": match["score"]
                    })
        
        store.close()
        return {"articles": articles}
        
    except Exception as e:
        logger.error(f"Article search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    
import uuid
from datetime import datetime, timedelta
from typing import Optional

from core.models import (DifficultyLevel, Flashcard, FlashcardFilters,
                         FlashcardRequest, FlashcardResponse, FlashcardStatus,
                         ReviewRequest, ReviewResponse, SourceType,
                         StudySession, StudySessionRequest, StudyStatistics)

# ============================================================================
# FLASHCARD ROUTES 
# ============================================================================
@app.post("/setup/flashcard-database")
async def setup_flashcard_database():
    """
    Bulletproof database setup that handles any existing state.
    """
    try:
        store = VectorStore()
        
        with store.pg_conn.cursor() as cursor:
            # Step 1: Check if flashcards table exists at all
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'flashcards'
                );
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                # Step 2: If table exists, drop it completely to avoid column conflicts
                print("Found existing flashcards table, dropping for clean recreation...")
                cursor.execute("DROP TABLE IF EXISTS study_sessions CASCADE")
                cursor.execute("DROP TABLE IF EXISTS user_progress CASCADE") 
                cursor.execute("DROP TABLE IF EXISTS flashcards CASCADE")
                print("Existing tables dropped successfully")
            
            # Step 3: Create fresh tables with complete schema
            print("Creating fresh flashcards table...")
            cursor.execute("""
                CREATE TABLE flashcards (
                    id VARCHAR(255) PRIMARY KEY,
                    category VARCHAR(255) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty VARCHAR(20) DEFAULT 'medium',
                    tags JSONB DEFAULT '[]'::jsonb,
                    status VARCHAR(20) DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_reviewed TIMESTAMP,
                    review_count INTEGER DEFAULT 0,
                    success_rate FLOAT DEFAULT 0.0,
                    next_review TIMESTAMP,
                    source_type VARCHAR(50),
                    news_date TIMESTAMP,
                    source_url TEXT,
                    context TEXT
                );
            """)
            
            print("Creating indexes...")
            cursor.execute("""
                CREATE INDEX idx_flashcards_category ON flashcards(category);
                CREATE INDEX idx_flashcards_status ON flashcards(status);
                CREATE INDEX idx_flashcards_next_review ON flashcards(next_review);
                CREATE INDEX idx_flashcards_difficulty ON flashcards(difficulty);
                CREATE INDEX idx_flashcards_news_date ON flashcards(news_date);
                CREATE INDEX idx_flashcards_source_type ON flashcards(source_type);
            """)
            
            print("Creating supporting tables...")
            cursor.execute("""
                CREATE TABLE study_sessions (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    card_id VARCHAR(255) NOT NULL,
                    correct BOOLEAN NOT NULL,
                    response_time_ms INTEGER,
                    difficulty_rating INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (card_id) REFERENCES flashcards(id) ON DELETE CASCADE
                );
                
                CREATE INDEX idx_study_sessions_card_id ON study_sessions(card_id);
                CREATE INDEX idx_study_sessions_session_id ON study_sessions(session_id);
                CREATE INDEX idx_study_sessions_created_at ON study_sessions(created_at);
                
                CREATE TABLE user_progress (
                    id SERIAL PRIMARY KEY,
                    date DATE DEFAULT CURRENT_DATE,
                    cards_studied INTEGER DEFAULT 0,
                    correct_answers INTEGER DEFAULT 0,
                    study_time_minutes INTEGER DEFAULT 0,
                    UNIQUE(date)
                );
            """)
            
            # Step 4: Commit all changes
            store.pg_conn.commit()
            print("Database setup completed successfully")
        
        store.close()
        
        return {
            "success": True,
            "message": "Flashcard database schema set up successfully with complete clean schema",
            "action_taken": "dropped_and_recreated" if table_exists else "created_new"
        }
        
    except Exception as e:
        print(f"Database setup error: {e}")
        logger.error(f"Database setup error: {e}")
        
        # Try to rollback if something went wrong
        try:
            store.pg_conn.rollback()
        except:
            pass
            
        raise HTTPException(status_code=500, detail=f"Database setup failed: {str(e)}")


# Also create a simple diagnostic endpoint to see what's currently in the database
@app.get("/setup/check-database")
async def check_database():
    """
    Simple check to see current database state.
    """
    try:
        store = VectorStore()
        
        with store.pg_conn.cursor() as cursor:
            # Check what tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            result = {
                "tables_found": tables,
                "flashcards_table_exists": "flashcards" in tables
            }
            
            # If flashcards table exists, check its columns
            if "flashcards" in tables:
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = 'flashcards'
                    ORDER BY ordinal_position;
                """)
                
                result["flashcards_columns"] = [
                    {
                        "name": row[0],
                        "type": row[1], 
                        "nullable": row[2],
                        "default": row[3]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Check row count
                cursor.execute("SELECT COUNT(*) FROM flashcards")
                result["flashcards_row_count"] = cursor.fetchone()[0]
        
        store.close()
        return result
        
    except Exception as e:
        logger.error(f"Database check error: {e}")
        return {
            "error": str(e),
            "tables_found": [],
            "flashcards_table_exists": False
        }@app.post("/flashcards/generate", response_model=FlashcardResponse)
async def generate_flashcards(request: FlashcardRequest) -> FlashcardResponse:
    """
    Generate flashcards based on topic and source type using the agent's tools.
    """
    try:
        # Use the agent's flashcard generation tool
        from agents.agent import TOOLS

        # Find the generate_flashcards_from_content tool
        generate_tool = next(
            (tool for tool in TOOLS if tool.name == "generate_flashcards_from_content"), 
            None
        )
        
        if not generate_tool:
            return FlashcardResponse(
                success=False,
                error="Flashcard generation tool not available"
            )
        
        result = generate_tool.invoke({
            "topic": request.topic,
            "source_type": request.source_type.value,
            "difficulty": request.difficulty.value,
            "count": request.count
        })
        
        flashcards_data = json.loads(result)
        
        if not flashcards_data.get("success"):
            return FlashcardResponse(
                success=False,
                error=flashcards_data.get("error", "Generation failed")
            )
        
        # Store flashcards in database with enhanced fields
        store = VectorStore()
        saved_cards = []
        
        try:
            with store.pg_conn.cursor() as cursor:
                for card_data in flashcards_data.get("flashcards", []):
                    # Create unique ID if not provided
                    card_id = card_data.get("id", str(uuid.uuid4()))
                    
                    cursor.execute("""
                        INSERT INTO flashcards 
                        (id, category, question, answer, difficulty, tags, status, created_at, 
                         success_rate, source_type, news_date, source_url, context)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            question = EXCLUDED.question,
                            answer = EXCLUDED.answer,
                            tags = EXCLUDED.tags,
                            context = EXCLUDED.context
                        RETURNING id
                    """, (
                        card_id,
                        card_data.get("category", request.topic or "General"), 
                        card_data["question"], 
                        card_data["answer"],
                        request.difficulty.value, 
                        json.dumps(card_data.get("tags", [])), 
                        FlashcardStatus.NEW.value,
                        datetime.now(), 
                        0.0,
                        request.source_type.value,
                        datetime.now() if request.source_type == SourceType.RECENT_NEWS else None,
                        card_data.get("source_url"),
                        card_data.get("context")
                    ))
                    
                    result_row = cursor.fetchone()
                    if result_row:  # Card was inserted or updated
                        flashcard = Flashcard(
                            id=card_id,
                            category=card_data.get("category", request.topic or "General"),
                            question=card_data["question"],
                            answer=card_data["answer"],
                            difficulty=request.difficulty,
                            tags=card_data.get("tags", []),
                            status=FlashcardStatus.NEW,
                            source_type=request.source_type,
                            context=card_data.get("context")
                        )
                        saved_cards.append(flashcard)
                
                store.pg_conn.commit()
        finally:
            store.close()
        
        return FlashcardResponse(
            success=True,
            flashcards=saved_cards,
            count=len(saved_cards),
            message=f"Generated and saved {len(saved_cards)} flashcards for topic: {request.topic}",
            topic=request.topic,
            source_type=request.source_type
        )
        
    except Exception as e:
        logger.error(f"Flashcard generation error: {e}")
        return FlashcardResponse(
            success=False,
            error=f"Generation failed: {str(e)}"
        )
@app.post("/flashcards/generate", response_model=FlashcardResponse)
async def generate_flashcards(request: FlashcardRequest) -> FlashcardResponse:
    """
    Generate flashcards based on topic and source type using the agent's tools.
    """
    try:
        # Use the agent's flashcard generation tool
        from agents.agent import TOOLS

        # Find the generate_flashcards_from_content tool
        generate_tool = next(
            (tool for tool in TOOLS if tool.name == "generate_flashcards_from_content"), 
            None
        )
        
        if not generate_tool:
            return FlashcardResponse(
                success=False,
                error="Flashcard generation tool not available"
            )
        
        result = generate_tool.invoke({
            "topic": request.topic,
            "source_type": request.source_type.value,
            "difficulty": request.difficulty.value,
            "count": request.count
        })
        
        flashcards_data = json.loads(result)
        
        if not flashcards_data.get("success"):
            return FlashcardResponse(
                success=False,
                error=flashcards_data.get("error", "Generation failed")
            )
        
        # Store flashcards in database with enhanced fields
        store = VectorStore()
        saved_cards = []
        
        try:
            with store.pg_conn.cursor() as cursor:
                for card_data in flashcards_data.get("flashcards", []):
                    # Create unique ID if not provided
                    card_id = card_data.get("id", str(uuid.uuid4()))
                    
                    cursor.execute("""
                        INSERT INTO flashcards 
                        (id, category, question, answer, difficulty, tags, status, created_at, 
                         success_rate, source_type, news_date, source_url, context)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            question = EXCLUDED.question,
                            answer = EXCLUDED.answer,
                            tags = EXCLUDED.tags,
                            context = EXCLUDED.context
                        RETURNING id
                    """, (
                        card_id,
                        card_data.get("category", request.topic or "General"), 
                        card_data["question"], 
                        card_data["answer"],
                        request.difficulty.value, 
                        json.dumps(card_data.get("tags", [])), 
                        FlashcardStatus.NEW.value,
                        datetime.now(), 
                        0.0,
                        request.source_type.value,
                        datetime.now() if request.source_type == SourceType.RECENT_NEWS else None,
                        card_data.get("source_url"),
                        card_data.get("context")
                    ))
                    
                    result_row = cursor.fetchone()
                    if result_row:  # Card was inserted or updated
                        flashcard = Flashcard(
                            id=card_id,
                            category=card_data.get("category", request.topic or "General"),
                            question=card_data["question"],
                            answer=card_data["answer"],
                            difficulty=request.difficulty,
                            tags=card_data.get("tags", []),
                            status=FlashcardStatus.NEW,
                            source_type=request.source_type,
                            context=card_data.get("context")
                        )
                        saved_cards.append(flashcard)
                
                store.pg_conn.commit()
        finally:
            store.close()
        
        return FlashcardResponse(
            success=True,
            flashcards=saved_cards,
            count=len(saved_cards),
            message=f"Generated and saved {len(saved_cards)} flashcards for topic: {request.topic}",
            topic=request.topic,
            source_type=request.source_type
        )
        
    except Exception as e:
        logger.error(f"Flashcard generation error: {e}")
        return FlashcardResponse(
            success=False,
            error=f"Generation failed: {str(e)}"
        )
@app.get("/flashcards", response_model=FlashcardResponse)
async def get_flashcards(
    category: Optional[str] = None,
    difficulty: Optional[DifficultyLevel] = None,
    status: Optional[FlashcardStatus] = None,
    due_for_review: bool = False,
    limit: int = 50,
    offset: int = 0,
    tags: Optional[str] = None  # Comma-separated tags
) -> FlashcardResponse:
    """
    Retrieve flashcards with advanced filters.
    """
    try:
        store = VectorStore()
        conditions = []
        params = []
        
        # Build dynamic query
        base_query = "SELECT * FROM flashcards WHERE 1=1"
        
        if category:
            conditions.append(" AND LOWER(category) LIKE LOWER(%s)")
            params.append(f"%{category}%")
        
        if difficulty:
            conditions.append(" AND difficulty = %s")
            params.append(difficulty.value)
        
        if status:
            conditions.append(" AND status = %s")
            params.append(status.value)
        
        if due_for_review:
            conditions.append(" AND (next_review IS NULL OR next_review <= %s)")
            params.append(datetime.now())
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            conditions.append(" AND tags ?| %s")
            params.append(tag_list)
        
        query = base_query + "".join(conditions) + " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        flashcards = []
        with store.pg_conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                flashcard = Flashcard(
                    id=row[0],
                    category=row[1],
                    question=row[2],
                    answer=row[3],
                    difficulty=DifficultyLevel(row[4]),
                    tags=json.loads(row[5]) if row[5] else [],
                    status=FlashcardStatus(row[6]),
                    created_at=row[7],
                    last_reviewed=row[8],
                    review_count=row[9],
                    success_rate=row[10],
                    next_review=row[11],
                    source_type=SourceType(row[12]) if row[12] else None,
                    news_date=row[13],
                    source_url=row[14],
                    context=row[15]
                )
                flashcards.append(flashcard)
        
        store.close()
        
        return FlashcardResponse(
            success=True,
            flashcards=flashcards,
            count=len(flashcards),
            message=f"Retrieved {len(flashcards)} flashcards"
        )
        
    except Exception as e:
        logger.error(f"Flashcard retrieval error: {e}")
        return FlashcardResponse(
            success=False,
            error=f"Retrieval failed: {str(e)}"
        )

@app.put("/flashcards/{card_id}/review", response_model=ReviewResponse)
async def review_flashcard(card_id: str, review: ReviewRequest) -> ReviewResponse:
    """
    Record flashcard review result with spaced repetition algorithm.
    """
    try:
        store = VectorStore()
        
        # Get current card data
        with store.pg_conn.cursor() as cursor:
            cursor.execute("SELECT * FROM flashcards WHERE id = %s", (card_id,))
            row = cursor.fetchone()
            
            if not row:
                return ReviewResponse(
                    success=False,
                    error="Flashcard not found"
                )
            
            current_success_rate = row[10] or 0.0
            review_count = row[9] or 0
            
            # Update success rate
            new_success_rate = ((current_success_rate * review_count) + (1 if review.correct else 0)) / (review_count + 1)
            
            # Calculate next review date using spaced repetition
            next_review = calculate_next_review_date(review.correct, review_count, new_success_rate)
            
            # Determine new status
            new_status = determine_card_status(review.correct, review_count, new_success_rate)
            
            # Update flashcard
            cursor.execute("""
                UPDATE flashcards 
                SET last_reviewed = %s, review_count = %s, success_rate = %s, 
                    next_review = %s, status = %s
                WHERE id = %s
            """, (
                datetime.now(), review_count + 1, new_success_rate, 
                next_review, new_status.value, card_id
            ))
            
            # Record study session
            cursor.execute("""
                INSERT INTO study_sessions 
                (session_id, card_id, correct, response_time_ms, difficulty_rating)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                card_id, review.correct, review.response_time_ms, review.difficulty_rating
            ))
            
            # Update daily progress
            cursor.execute("""
                INSERT INTO user_progress (date, cards_studied, correct_answers)
                VALUES (CURRENT_DATE, 1, %s)
                ON CONFLICT (date) DO UPDATE SET
                    cards_studied = user_progress.cards_studied + 1,
                    correct_answers = user_progress.correct_answers + %s
            """, (1 if review.correct else 0, 1 if review.correct else 0))
            
            store.pg_conn.commit()
        
        store.close()
        
        return ReviewResponse(
            success=True,
            message="Review recorded successfully",
            next_review=next_review,
            new_status=new_status,
            success_rate=new_success_rate
        )
        
    except Exception as e:
        logger.error(f"Review recording error: {e}")
        return ReviewResponse(
            success=False,
            error=f"Review recording failed: {str(e)}"
        )

@app.post("/flashcards/study-session", response_model=StudySession)
async def start_study_session(request: StudySessionRequest) -> StudySession:
    """
    Start a new study session with optimized card selection.
    """
    try:
        store = VectorStore()
        conditions = []
        params = []
        
        # Build query for session cards
        query = """
            SELECT * FROM flashcards 
            WHERE 1=1
        """
        
        if request.topics:
            topic_conditions = " OR ".join(["LOWER(category) LIKE LOWER(%s)" for _ in request.topics])
            conditions.append(f" AND ({topic_conditions})")
            params.extend([f"%{topic}%" for topic in request.topics])
        
        if request.difficulty:
            conditions.append(" AND difficulty = %s")
            params.append(request.difficulty.value)
        
        # Priority: due cards first, then new cards
        if request.include_due and request.include_new:
            conditions.append(" AND (next_review IS NULL OR next_review <= %s OR status = 'new')")
            params.append(datetime.now())
        elif request.include_due:
            conditions.append(" AND next_review <= %s")
            params.append(datetime.now())
        elif request.include_new:
            conditions.append(" AND status = 'new'")
        
        query += "".join(conditions)
        query += " ORDER BY CASE WHEN next_review <= %s THEN 0 ELSE 1 END, next_review ASC LIMIT %s"
        params.extend([datetime.now(), request.max_cards])
        
        flashcards = []
        with store.pg_conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                flashcard = Flashcard(
                    id=row[0],
                    category=row[1],
                    question=row[2],
                    answer=row[3],
                    difficulty=DifficultyLevel(row[4]),
                    tags=json.loads(row[5]) if row[5] else [],
                    status=FlashcardStatus(row[6])
                )
                flashcards.append(flashcard)
        
        store.close()
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        return StudySession(
            session_id=session_id,
            cards=flashcards,
            total_cards=len(flashcards),
            topics=request.topics or ["all"]
        )
        
    except Exception as e:
        logger.error(f"Study session error: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.get("/flashcards/stats", response_model=StudyStatistics)
async def get_study_stats() -> StudyStatistics:
    """
    Get comprehensive study statistics.
    """
    try:
        store = VectorStore()
        stats = StudyStatistics()
        
        with store.pg_conn.cursor() as cursor:
            # Basic card counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN next_review <= %s THEN 1 END) as due,
                    COUNT(CASE WHEN status = 'new' THEN 1 END) as new,
                    COUNT(CASE WHEN status = 'learning' THEN 1 END) as learning,
                    COUNT(CASE WHEN status = 'known' THEN 1 END) as known,
                    AVG(success_rate) as avg_success_rate
                FROM flashcards
            """, (datetime.now(),))
            
            row = cursor.fetchone()
            if row:
                stats.total_cards = row[0] or 0
                stats.cards_due = row[1] or 0
                stats.cards_new = row[2] or 0
                stats.cards_learning = row[3] or 0
                stats.cards_known = row[4] or 0
                stats.overall_success_rate = round(row[5] or 0.0, 2)
            
            # Today's activity
            cursor.execute("""
                SELECT cards_studied, correct_answers 
                FROM user_progress 
                WHERE date = CURRENT_DATE
            """)
            
            today_row = cursor.fetchone()
            if today_row:
                stats.cards_studied_today = today_row[0] or 0
            
            # Category breakdown
            cursor.execute("""
                SELECT category, COUNT(*) 
                FROM flashcards 
                GROUP BY category 
                ORDER BY COUNT(*) DESC
            """)
            
            for cat_row in cursor.fetchall():
                stats.categories[cat_row[0]] = cat_row[1]
            
            # Recent activity (last 7 days)
            cursor.execute("""
                SELECT date, cards_studied, correct_answers
                FROM user_progress 
                WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY date DESC
            """)
            
            for activity_row in cursor.fetchall():
                stats.recent_activity.append({
                    "date": str(activity_row[0]),
                    "cards_studied": activity_row[1],
                    "correct_answers": activity_row[2],
                    "accuracy": round((activity_row[2] / activity_row[1]) * 100, 1) if activity_row[1] > 0 else 0
                })
        
        store.close()
        return stats
        
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

@app.delete("/flashcards/{card_id}")
async def delete_flashcard(card_id: str):
    """Delete a flashcard."""
    try:
        store = VectorStore()
        
        with store.pg_conn.cursor() as cursor:
            cursor.execute("DELETE FROM flashcards WHERE id = %s", (card_id,))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Flashcard not found")
            store.pg_conn.commit()
        
        store.close()
        return {"success": True, "message": "Flashcard deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_next_review_date(correct: bool, review_count: int, success_rate: float) -> datetime:
    """Calculate next review date using spaced repetition algorithm."""
    base_interval = 1  # days
    
    if correct:
        # Successful review - increase interval
        if review_count == 0:
            interval = 1
        elif review_count == 1:
            interval = 3
        else:
            # SM-2 inspired algorithm
            ease_factor = 2.5 + (success_rate - 0.8) * 0.5
            ease_factor = max(1.3, min(2.5, ease_factor))
            interval = max(1, int(base_interval * (ease_factor ** review_count)))
    else:
        # Failed review - reset to shorter interval
        interval = max(1, review_count // 3)
    
    return datetime.now() + timedelta(days=interval)

def determine_card_status(correct: bool, review_count: int, success_rate: float) -> FlashcardStatus:
    """Determine card status based on performance."""
    if review_count < 2:
        return FlashcardStatus.NEW
    elif success_rate >= 0.8 and review_count >= 5:
        return FlashcardStatus.KNOWN
    elif success_rate >= 0.6:
        return FlashcardStatus.LEARNING
    else:
        return FlashcardStatus.REVIEW
# ============================================================================
# STREAMING FUNCTIONS
# ============================================================================

async def stream_agent_response(messages: List):
    """
    Stream the agent's response using Server-Sent Events.
    
    Args:
        messages: List of chat messages
        
    Yields:
        SSE formatted responses
    """
    try:
        # Initialize state
        state = {"messages": messages}
        
        # Stream the graph execution
        async for chunk in agent_graph.astream(state):
            for node_name, node_output in chunk.items():
                if "messages" in node_output and node_output["messages"]:
                    latest_message = node_output["messages"][-1]
                    
                    # Handle AI messages (responses)
                    if isinstance(latest_message, AIMessage) and latest_message.content:
                        # Stream content word by word for better UX
                        words = latest_message.content.split()
                        for i, word in enumerate(words):
                            if i == 0:
                                yield f"data: {json.dumps({'type': 'content', 'data': word})}\n\n"
                            else:
                                yield f"data: {json.dumps({'type': 'content', 'data': f' {word}'})}\n\n"
                            
                            # Small delay for smoother streaming
                            await asyncio.sleep(0.01)
                    
                    # Handle tool calls
                    elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
                        for tool_call in latest_message.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'data': {'name': tool_call['name'], 'args': tool_call.get('args', {})}})}\n\n"
        
        # Signal completion
        yield f"data: {json.dumps({'type': 'done', 'data': None})}\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

async def get_complete_response(messages: List) -> str:
    """
    Get complete response without streaming.
    
    Args:
        messages: List of chat messages
        
    Returns:
        Complete response string
    """
    try:
        state = {"messages": messages}
        result = await agent_graph.ainvoke(state)
        
        if result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                return last_message.content
        
        return "I apologize, but I couldn't generate a response."
        
    except Exception as e:
        logger.error(f"Complete response error: {e}")
        return f"Error: {str(e)}"


# ============================================================================
# TIMELINE ROUTES
# ============================================================================

@app.post("/api/v1/timeline/", response_model=TimelineResponse)
async def generate_timeline(request: TimelineRequest):
    """
    Generate timeline for a specific entity based on stored articles.
    Automatically fetches news if no data exists for the entity.
    """
    try:
        logger.info(f"Generating timeline for entity: {request.entity_name}")
        store = VectorStore()
        
        if not store.pg_conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Calculate date range
        end_date = request.end_date or datetime.now()
        
        if request.time_range == "7d":
            start_date = end_date - timedelta(days=7)
        elif request.time_range == "30d":
            start_date = end_date - timedelta(days=30)
        elif request.time_range == "90d":
            start_date = end_date - timedelta(days=90)
        elif request.time_range == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = request.start_date or (end_date - timedelta(days=30))
        
        # First, check if we have existing data for this entity
        with store.pg_conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE (
                    LOWER(title) LIKE LOWER(%s) OR 
                    LOWER(content) LIKE LOWER(%s) OR 
                    LOWER(summary) LIKE LOWER(%s) OR
                    %s = ANY(entities)
                )
                AND published_date BETWEEN %s AND %s
            """, (
                f"%{request.entity_name}%",
                f"%{request.entity_name}%", 
                f"%{request.entity_name}%",
                request.entity_name,
                start_date,
                end_date
            ))
            
            result = cursor.fetchone()
            existing_count = result[0] if result else 0
        
        # If we have less than 3 articles, fetch more news
        if existing_count < 3:
            logger.info(f"Only {existing_count} articles found for {request.entity_name}, fetching fresh news...")
            
            try:
                # Fetch fresh news for this entity
                news_data = fetch_gnews(request.entity_name)
                
                if news_data and 'articles' in news_data and news_data['articles']:
                    logger.info(f"Fetched {len(news_data['articles'])} new articles")
                    
                    # Process articles immediately (inline processing)
                    for article_data in news_data['articles']:
                        try:
                            # Create NewsArticle model
                            article = NewsArticle(
                                id=str(hash(article_data.get('url', ''))),
                                title=article_data.get('title', ''),
                                content=article_data.get('content', ''),
                                summary=article_data.get('description', ''),
                                url=article_data.get('url', ''),
                                source=article_data.get('source', {}).get('name', 'Unknown'),
                                author=article_data.get('author'),
                                published_date=datetime.fromisoformat(
                                    article_data.get('publishedAt', datetime.now().isoformat()).replace('Z', '+00:00')
                                ),
                                categories=[],
                                entities=[request.entity_name],  # Tag with searched entity
                                relevance_score=0.8  # High relevance since it was specifically searched
                            )
                            
                            # Store the article
                            store.insert_article(article)
                            logger.info(f"Stored article: {article.title[:50]}...")
                            
                        except Exception as e:
                            logger.error(f"Error processing article: {e}")
                            continue
                else:
                    logger.warning(f"No news articles found for {request.entity_name}")
                    
            except Exception as e:
                logger.error(f"Error fetching news for {request.entity_name}: {e}")
                # Continue with existing data even if news fetch fails
        
        # Now search for articles mentioning the entity (including newly fetched ones)
        with store.pg_conn.cursor() as cursor:
            # Search in title, content, summary, and entities array
            cursor.execute("""
                SELECT id, title, content, summary, url, source, author, published_date, 
                       categories, entities, relevance_score
                FROM articles 
                WHERE (
                    LOWER(title) LIKE LOWER(%s) OR 
                    LOWER(content) LIKE LOWER(%s) OR 
                    LOWER(summary) LIKE LOWER(%s) OR
                    %s = ANY(entities)
                )
                AND published_date BETWEEN %s AND %s
                ORDER BY published_date DESC
                LIMIT %s
            """, (
                f"%{request.entity_name}%",
                f"%{request.entity_name}%", 
                f"%{request.entity_name}%",
                request.entity_name,
                start_date,
                end_date,
                request.limit
            ))
            
            articles = cursor.fetchall()
        
        # Convert articles to timeline events
        events = []
        entity_mentions = {}
        
        if articles:
            # Enhanced story progression with clustering
            events = _create_story_timeline(articles, request.entity_name)
            
            # Track entity mentions from all events
            for event in events:
                article_entities = event.entities or []
                for entity in article_entities:
                    if entity.lower() != request.entity_name.lower():
                        if entity not in entity_mentions:
                            entity_mentions[entity] = {
                                "count": 0,
                                "relevance_sum": 0,
                                "first_seen": event.date,
                                "last_seen": event.date
                            }
                        entity_mentions[entity]["count"] += 1
                        entity_mentions[entity]["relevance_sum"] += event.relevance_score
                        entity_mentions[entity]["last_seen"] = max(
                            entity_mentions[entity]["last_seen"], event.date
                        )
        
        # Calculate enhanced statistics with clustering info
        story_clusters = {}
        for event in events:
            cluster_id = event.metadata.get("story_cluster", -1)
            if cluster_id not in story_clusters:
                story_clusters[cluster_id] = 0
            story_clusters[cluster_id] += 1
        
        statistics = {
            "total_events": len(events),
            "articles_analyzed": len(articles),
            "time_span_days": (end_date - start_date).days,
            "average_relevance": sum(e.relevance_score for e in events) / len(events) if events else 0,
            "fresh_data_fetched": existing_count < 3,  # Indicate if we fetched fresh data
            "story_threads": int(len([c for c in story_clusters.keys() if c >= 0])),
            "standalone_articles": int(story_clusters.get(-1, 0)),
            "clustering_summary": {str(int(k)): int(v) for k, v in story_clusters.items()}
        }
        
        # Get related entities
        related_entities = []
        for entity, data in sorted(entity_mentions.items(), 
                                 key=lambda x: x[1]["count"], reverse=True)[:10]:
            related_entities.append({
                "name": entity,
                "mentions": data["count"],
                "avg_relevance": data["relevance_sum"] / data["count"],
                "first_mentioned": data["first_seen"].isoformat(),
                "last_mentioned": data["last_seen"].isoformat()
            })
        
        store.close()
        
        return TimelineResponse(
            success=True,
            entity_name=request.entity_name,
            events=events,
            total_events=len(events),
            time_range=request.time_range,
            start_date=start_date,
            end_date=end_date,
            statistics=statistics,
            related_entities=related_entities
        )
        
    except Exception as e:
        logger.error(f"Timeline generation error: {e}")
        return TimelineResponse(
            success=False,
            entity_name=request.entity_name,
            events=[],
            total_events=0,
            time_range=request.time_range,
            start_date=datetime.now(),
            end_date=datetime.now(),
            error=str(e)
        )

@app.get("/api/v1/timeline/{entity_name}")
async def get_entity_timeline(
    entity_name: str,
    time_range: str = "30d"
):
    """
    Get timeline for a specific entity with query parameters.
    """
    request = TimelineRequest(
        entity_name=entity_name,
        time_range=time_range
    )
    return await generate_timeline(request)

@app.get("/api/v1/timeline/{entity_name}/summary")
async def get_timeline_summary(entity_name: str, time_range: str = "30d"):
    """
    Get summarized timeline data for an entity.
    """
    try:
        timeline_response = await get_entity_timeline(entity_name, time_range)
        
        if not timeline_response.success:
            raise HTTPException(status_code=500, detail=timeline_response.error)
        
        events = timeline_response.events
        
        # Generate key developments
        key_developments = []
        if events:
            # Get highest relevance events
            top_events = sorted(events, key=lambda x: x.relevance_score, reverse=True)[:5]
            key_developments = [event.title for event in top_events]
        
        # Calculate sentiment trend (daily) - simplified since we removed sentiment analysis
        sentiment_trend = []
        if events:
            from collections import defaultdict
            daily_events = defaultdict(int)
            
            for event in events:
                date_key = event.date.strftime("%Y-%m-%d")
                daily_events[date_key] += 1
            
            sentiment_trend = [
                {
                    "date": date,
                    "events": count
                }
                for date, count in sorted(daily_events.items())
            ]
        
        # Get trending topics from categories
        trending_topics = []
        if events:
            from collections import Counter
            all_categories = []
            for event in events:
                all_categories.extend(event.metadata.get("categories", []))
            trending_topics = [topic for topic, count in Counter(all_categories).most_common(10)]
        
        return TimelineSummary(
            entity_name=entity_name,
            total_events=len(events),
            time_span_days=timeline_response.statistics.get("time_span_days", 0),
            key_developments=key_developments,
            sentiment_trend=sentiment_trend,
            related_entities=[
                EntityMention(
                    entity_name=entity["name"],
                    mention_count=entity["mentions"],
                    relevance_score=entity["avg_relevance"],
                    first_mentioned=datetime.fromisoformat(entity["first_mentioned"]),
                    last_mentioned=datetime.fromisoformat(entity["last_mentioned"]),
                    sentiment_distribution={}
                )
                for entity in timeline_response.related_entities
            ],
            trending_topics=trending_topics
        )
        
    except Exception as e:
        logger.error(f"Timeline summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate timeline summary: {str(e)}")

@app.get("/api/v1/timeline/{entity_name}/related")
async def get_related_entities_timeline(entity_name: str, time_range: str = "30d"):
    """
    Get timeline data for entities related to the specified entity.
    """
    try:
        # First get the main entity timeline to find related entities
        main_timeline = await get_entity_timeline(entity_name, time_range)
        
        if not main_timeline.success:
            raise HTTPException(status_code=500, detail=main_timeline.error)
        
        # Get top 5 related entities
        related_entities = main_timeline.related_entities[:5]
        
        # Generate timelines for each related entity
        related_timelines = []
        for entity_data in related_entities:
            related_name = entity_data["name"]
            related_timeline = await get_entity_timeline(related_name, time_range)
            
            if related_timeline.success and related_timeline.events:
                related_timelines.append({
                    "entity_name": related_name,
                    "events": related_timeline.events[:10],  # Limit to 10 events per entity
                    "total_events": related_timeline.total_events,
                    "relevance_to_main": entity_data["avg_relevance"]
                })
        
        return {
            "main_entity": entity_name,
            "related_timelines": related_timelines,
            "total_related_entities": len(related_timelines)
        }
        
    except Exception as e:
        logger.error(f"Related entities timeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get related entities timeline: {str(e)}")

@app.get("/api/v1/timeline/events/trending")
async def get_trending_events(time_range: str = "7d", limit: int = 20):
    """
    Get trending events across all entities.
    """
    try:
        store = VectorStore()
        
        if not store.pg_conn:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Calculate date range
        end_date = datetime.now()
        if time_range == "7d":
            start_date = end_date - timedelta(days=7)
        elif time_range == "30d":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=7)
        
        # Get recent articles with high relevance scores
        with store.pg_conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, title, content, summary, url, source, author, published_date, 
                       categories, entities, relevance_score
                FROM articles 
                WHERE published_date BETWEEN %s AND %s
                AND relevance_score > 0.5
                ORDER BY relevance_score DESC, published_date DESC
                LIMIT %s
            """, (start_date, end_date, limit))
            
            articles = cursor.fetchall()
        
        # Convert to timeline events
        trending_events = []
        for article in articles:
            relevance = article[10] or 0.5
            
            event = TimelineEvent(
                id=f"trending_{article[0]}",
                title=article[1],
                description=article[3] or article[2][:200] + "...",
                date=article[7],
                event_type=EventType.NEWS,
                sentiment=SentimentType.NEUTRAL,  # Remove sentiment analysis
                relevance_score=relevance,
                entities=article[9] or [],
                source_article_id=str(article[0]),
                source_url=article[4],
                metadata={
                    "source": article[5],
                    "author": article[6],
                    "categories": article[8] or [],
                    "source_credibility": _calculate_source_credibility(article[5])
                }
            )
            trending_events.append(event)
        
        store.close()
        
        return {
            "success": True,
            "events": trending_events,
            "total_events": len(trending_events),
            "time_range": time_range,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Trending events error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trending events: {str(e)}")

# ============================================================================
# TOP HEADLINES ROUTES
# ============================================================================

@app.get("/api/v1/headlines/top")
async def get_top_headlines(
    category: str = "general", 
    lang: str = "en", 
    country: str = "us", 
    max_articles: int = 8
):
    """
    Get top headlines from GNews API.
    
    Args:
        category: Category of news (general, world, nation, business, technology, entertainment, sports, science, health)
        lang: Language code (en, es, fr, etc.)
        country: Country code (us, uk, ca, etc.)
        max_articles: Maximum number of articles to return (1-100)
        
    Returns:
        Top headlines with source credibility information
    """
    try:
        # Validate parameters
        max_articles = min(max(max_articles, 1), 100)
        
        # Available categories based on GNews API
        valid_categories = [
            "general", "world", "nation", "business", "technology", 
            "entertainment", "sports", "science", "health"
        ]
        
        if category not in valid_categories:
            category = "general"
        
        # Fetch headlines from GNews
        import requests
        from core.config import settings
        
        url = f"https://gnews.io/api/v4/top-headlines"
        params = {
            "category": category,
            "lang": lang,
            "country": country,
            "max": max_articles,
            "apikey": settings.GNEWS_API_KEY
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if "articles" not in data:
            return {
                "success": False,
                "error": "No articles found",
                "headlines": []
            }
        
        # Transform articles to include credibility information
        headlines = []
        for article in data["articles"]:
            source_credibility = _calculate_source_credibility(
                article.get("source", {}).get("name", "")
            )
            
            headline = {
                "id": article.get("id", ""),
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "image": article.get("image", ""),
                "published_at": article.get("publishedAt", ""),
                "source": {
                    "name": article.get("source", {}).get("name", ""),
                    "url": article.get("source", {}).get("url", ""),
                    "credibility": source_credibility
                },
                "category": category,
                "country": country,
                "language": lang
            }
            headlines.append(headline)
        
        return {
            "success": True,
            "total_articles": data.get("totalArticles", len(headlines)),
            "category": category,
            "country": country,
            "language": lang,
            "headlines": headlines
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"GNews API error: {e}")
        return {
            "success": False,
            "error": f"Failed to fetch headlines: {str(e)}",
            "headlines": []
        }
    except Exception as e:
        logger.error(f"Top headlines error: {e}")
        return {
            "success": False,
            "error": f"Internal server error: {str(e)}",
            "headlines": []
        }

@app.get("/api/v1/headlines/categories")
async def get_available_categories():
    """
    Get available news categories for top headlines.
    """
    return {
        "categories": [
            {"id": "general", "label": "General", "description": "General news and current events"},
            {"id": "world", "label": "World", "description": "International news and global events"},
            {"id": "nation", "label": "Nation", "description": "National news and domestic events"},
            {"id": "business", "label": "Business", "description": "Business news, finance, and markets"},
            {"id": "technology", "label": "Technology", "description": "Tech news, innovations, and digital trends"},
            {"id": "entertainment", "label": "Entertainment", "description": "Entertainment news, movies, music, and celebrities"},
            {"id": "sports", "label": "Sports", "description": "Sports news, scores, and athlete updates"},
            {"id": "science", "label": "Science", "description": "Scientific discoveries and research"},
            {"id": "health", "label": "Health", "description": "Health news, medical breakthroughs, and wellness"}
        ]
    }

# Helper functions for timeline processing
def _calculate_source_credibility(source_name: str) -> dict:
    """
    Calculate source credibility and political leaning indicators based on comprehensive journalism standards.
    Research sources: Reuters Institute Digital News Report 2024, Wikipedia circulation data,
    Pew Research Center Political Polarization & Media Habits study, and major international news agency standards.
    
    Returns dictionary with credibility score, political leaning, and metadata.
    """
    if not source_name:
        return {"score": 0.5, "tier": "unknown", "category": "unknown", "political_leaning": "center"}
    
    source_lower = source_name.lower()
    
    # TIER 1: PREMIER INTERNATIONAL WIRE SERVICES & MAJOR NEWSPAPERS (Score: 0.95)
    # International wire services, papers of record, high circulation dailies
    tier1_sources = [
        # Major International Wire Services (Primary sources for global news)
        'reuters', 'associated press', 'ap news', 'agence france-presse', 'afp',
        'bbc news', 'bbc', 'bloomberg news', 'bloomberg',
        
        # Papers of Record & High Circulation Newspapers
        'the new york times', 'nytimes', 'wall street journal', 'wsj',
        'the washington post', 'washingtonpost', 'the guardian', 'guardian',
        'financial times', 'ft.com', 'the times', 'times of london',
        'usa today', 'usatoday',  # #3 circulation in US
        
        # Major International Newspapers
        'the telegraph', 'telegraph.co.uk', 'the independent', 'independent.co.uk',
        'daily mail', 'dailymail.co.uk', 'the sun', 'thesun.co.uk',
        'le monde', 'lemonde.fr', 'le figaro', 'lefigaro.fr',
        'frankfurter allgemeine', 'faz.net', 'der spiegel', 'spiegel.de',
        'süddeutsche zeitung', 'sueddeutsche.de', 'die zeit', 'zeit.de',
        'corriere della sera', 'corriere.it', 'la repubblica', 'repubblica.it',
        'la gazzetta dello sport', 'gazzetta.it', 'el país', 'elpais.com',
        'el mundo', 'elmundo.es', 'la vanguardia', 'lavanguardia.com',
        'yomiuri shimbun', 'yomiuri.co.jp', 'asahi shimbun', 'asahi.com',
        'mainichi shimbun', 'mainichi.jp', 'nikkei', 'nikkei.com',
        
        # Major TV News Networks (Established)
        'cnn', 'cnn.com', 'fox news', 'foxnews.com', 'abc news', 'abcnews.go.com',
        'cbs news', 'cbsnews.com', 'nbc news', 'nbcnews.com', 'pbs', 'pbs.org',
        'npr', 'npr.org'
    ]
    
    # TIER 2: ESTABLISHED NATIONAL & REGIONAL NEWS ORGANIZATIONS (Score: 0.8)
    # National news agencies, established regional papers, specialty publications
    tier2_sources = [
        # National News Agencies by Country
        'canadian press', 'cp.ca', 'australian associated press', 'aap',
        'press association', 'pa media', 'deutsche presse-agentur', 'dpa',
        'agencia efe', 'efe.com', 'ansa', 'ansa.it', 'kyodo news', 'kyodo.co.jp',
        'yonhap', 'yonhapnews.co.kr', 'xinhua', 'xinhuanet.com',
        'tass', 'tass.ru', 'interfax', 'interfax.ru',
        'press trust of india', 'pti', 'united news of india', 'uni',
        
        # Major Regional & National Papers
        'chicago tribune', 'chicagotribune.com', 'los angeles times', 'latimes.com',
        'boston globe', 'bostonglobe.com', 'denver post', 'denverpost.com',
        'san francisco chronicle', 'sfchronicle.com', 'atlanta journal-constitution',
        'philadelphia inquirer', 'inquirer.com', 'miami herald', 'miamiherald.com',
        'seattle times', 'seattletimes.com', 'dallas morning news', 'dallasnews.com',
        'houston chronicle', 'houstonchronicle.com',
        
        # Major International Outlets
        'sky news', 'skynews.com', 'itv news', 'itv.com', 'channel 4 news',
        'france 24', 'france24.com', 'dw', 'dw.com', 'rt', 'rt.com',
        'al jazeera', 'aljazeera.com', 'cbc', 'cbc.ca', 'ctv news', 'ctvnews.ca',
        'global news', 'globalnews.ca', 'nine news', 'nine.com.au',
        'abc australia', 'abc.net.au', 'sbs news', 'sbs.com.au',
        
        # Business & Financial Publications
        'cnbc', 'cnbc.com', 'marketwatch', 'marketwatch.com', 'forbes', 'forbes.com',
        'fortune', 'fortune.com', 'business insider', 'businessinsider.com',
        'the economist', 'economist.com', 'harvard business review', 'hbr.org',
        
        # Quality Magazines & Weeklies
        'time', 'time.com', 'newsweek', 'newsweek.com', 'the atlantic', 'theatlantic.com',
        'the new yorker', 'newyorker.com', 'harper\'s magazine', 'harpers.org',
        'the nation', 'thenation.com', 'national review', 'nationalreview.com',
        'foreign affairs', 'foreignaffairs.com', 'foreign policy', 'foreignpolicy.com'
    ]
    
    # TIER 3: CREDIBLE SPECIALIZED & EMERGING SOURCES (Score: 0.65)
    # Tech publications, scientific journals, regional outlets, digital-first media
    tier3_sources = [
        # Technology Publications
        'techcrunch', 'techcrunch.com', 'wired', 'wired.com', 'the verge', 'theverge.com',
        'ars technica', 'arstechnica.com', 'engadget', 'engadget.com',
        'recode', 'recode.net', 'mashable', 'mashable.com', 'gizmodo', 'gizmodo.com',
        'cnet', 'cnet.com', 'zdnet', 'zdnet.com', 'computerworld', 'computerworld.com',
        
        # Scientific & Academic Publications
        'scientific american', 'scientificamerican.com', 'nature', 'nature.com',
        'science', 'science.org', 'new scientist', 'newscientist.com',
        'ieee spectrum', 'spectrum.ieee.org', 'mit technology review', 'technologyreview.com',
        
        # Digital-First Quality News
        'vox', 'vox.com', 'buzzfeed news', 'buzzfeednews.com', 'huffpost', 'huffpost.com',
        'politico', 'politico.com', 'the hill', 'thehill.com', 'axios', 'axios.com',
        'propublica', 'propublica.org', 'the intercept', 'theintercept.com',
        
        # Entertainment & Culture
        'variety', 'variety.com', 'hollywood reporter', 'hollywoodreporter.com',
        'entertainment weekly', 'ew.com', 'rolling stone', 'rollingstone.com',
        
        # Sports Publications
        'espn', 'espn.com', 'sports illustrated', 'si.com', 'the athletic', 'theathletic.com',
        'bbc sport', 'skysports', 'espn.co.uk',
        
        # Aggregation Services
        'yahoo news', 'yahoo.com', 'msn news', 'msn.com', 'google news', 'news.google.com',
        'apple news', 'news.apple.com'
    ]
    
    # POLITICAL LEANING CLASSIFICATIONS
    # Based on Pew Research Center Political Polarization & Media Habits study and media bias research
    
    # LEFT-LEANING SOURCES (Liberal/Progressive)
    left_sources = [
        # Consistent Liberal Sources (Pew Research)
        'msnbc', 'msnbc.com', 'the new yorker', 'newyorker.com', 'slate', 'slate.com',
        'mother jones', 'motherjones.com', 'the nation', 'thenation.com',
        'democracy now', 'democracynow.org', 'the intercept', 'theintercept.com',
        'jacobin', 'jacobinmag.com', 'in these times', 'inthesetimes.com',
        'the progressive', 'progressive.org', 'common dreams', 'commondreams.org',
        
        # Leans Left Sources
        'huffpost', 'huffpost.com', 'the daily beast', 'thedailybeast.com',
        'buzzfeed news', 'buzzfeednews.com', 'vox', 'vox.com', 'salon', 'salon.com',
        'alternet', 'alternet.org', 'raw story', 'rawstory.com',
        'the guardian', 'guardian.com', 'the independent', 'independent.co.uk',
        'washington post', 'washingtonpost.com', 'new york times', 'nytimes.com',
        'cnn', 'cnn.com', 'npr', 'npr.org', 'pbs', 'pbs.org',
        'la times', 'latimes.com', 'boston globe', 'bostonglobe.com'
    ]
    
    # RIGHT-LEANING SOURCES (Conservative)  
    right_sources = [
        # Consistent Conservative Sources (Pew Research)
        'fox news', 'foxnews.com', 'talk radio news', 'rush limbaugh', 'sean hannity',
        'glenn beck', 'breitbart', 'breitbart.com', 'daily wire', 'dailywire.com',
        'townhall', 'townhall.com', 'redstate', 'redstate.com', 'hot air', 'hotair.com',
        'pj media', 'pjmedia.com', 'the federalist', 'thefederalist.com',
        'american thinker', 'americanthinker.com', 'newsmax', 'newsmax.com',
        'one america news', 'oann', 'the blaze', 'theblaze.com',
        
        # Leans Right Sources
        'wall street journal', 'wsj.com', 'washington examiner', 'washingtonexaminer.com',
        'new york post', 'nypost.com', 'daily mail', 'dailymail.co.uk',
        'fox business', 'foxbusiness.com', 'national review', 'nationalreview.com',
        'weekly standard', 'weeklystandard.com', 'reason', 'reason.com',
        'washington times', 'washingtontimes.com', 'american spectator', 'spectator.org',
        'drudge report', 'drudgereport.com'
    ]
    
    # CENTER SOURCES (Neutral/Mixed Audience)
    center_sources = [
        # Wire Services (Neutral by design)
        'reuters', 'associated press', 'ap news', 'agence france-presse', 'afp',
        'bloomberg', 'bloomberg.com',
        
        # Mixed Audience Sources (Pew Research)
        'usa today', 'usatoday.com', 'abc news', 'abcnews.go.com',
        'cbs news', 'cbsnews.com', 'nbc news', 'nbcnews.com',
        'bbc', 'bbc.com', 'bbc news', 'time', 'time.com',
        'newsweek', 'newsweek.com', 'the economist', 'economist.com',
        'financial times', 'ft.com', 'christian science monitor', 'csmonitor.com',
        'yahoo news', 'yahoo.com', 'google news', 'news.google.com',
        'msn news', 'msn.com', 'local tv', 'local news'
    ]
    
    def matches_source(source_list, source_lower):
        """Check if source matches any in the list with partial matching"""
        for source in source_list:
            # Direct match or substring match
            if source in source_lower or source_lower in source:
                return True
            # Domain matching (remove common prefixes/suffixes)
            source_clean = source.replace('the ', '').replace('.com', '').replace('.co.uk', '').replace('.org', '')
            source_lower_clean = source_lower.replace('the ', '').replace('.com', '').replace('.co.uk', '').replace('.org', '')
            if source_clean in source_lower_clean or source_lower_clean in source_clean:
                return True
        return False
    
    def determine_political_leaning(source_lower):
        """Determine political leaning based on source classification"""
        if matches_source(left_sources, source_lower):
            return "left"
        elif matches_source(right_sources, source_lower):
            return "right"
        elif matches_source(center_sources, source_lower):
            return "center"
        else:
            return "center"  # Default to center for unknown sources
    
    # Determine political leaning
    political_leaning = determine_political_leaning(source_lower)
    
    # Check against credibility tiers
    if matches_source(tier1_sources, source_lower):
        return {
            "score": 0.95, 
            "tier": "tier1", 
            "category": "premier_international",
            "political_leaning": political_leaning
        }
    
    if matches_source(tier2_sources, source_lower):
        return {
            "score": 0.8, 
            "tier": "tier2", 
            "category": "established_national",
            "political_leaning": political_leaning
        }
    
    if matches_source(tier3_sources, source_lower):
        return {
            "score": 0.65, 
            "tier": "tier3", 
            "category": "credible_specialized",
            "political_leaning": political_leaning
        }
    
    # Default for unknown sources - slightly lower to encourage known sources
    return {
        "score": 0.55, 
        "tier": "unrated", 
        "category": "unknown",
        "political_leaning": political_leaning
    }

def _calculate_relevance(article, entity_name: str) -> float:
    """
    Calculate relevance score for an article in relation to an entity.
    """
    title = article[1].lower()
    content = (article[2] or "").lower()
    summary = (article[3] or "").lower()
    entities = article[9] or []
    
    entity_lower = entity_name.lower()
    score = 0.0
    
    # Direct mention in title (highest weight)
    if entity_lower in title:
        score += 0.5
    
    # Direct mention in summary
    if entity_lower in summary:
        score += 0.3
    
    # Entity in entities list
    if entity_name in entities or entity_lower in [e.lower() for e in entities]:
        score += 0.4
    
    # Mention in content (lower weight)
    if entity_lower in content:
        score += 0.2
    
    # Use existing relevance score if available
    if article[10]:
        score = max(score, article[10])
    
    return min(score, 1.0)  # Cap at 1.0


def _ensure_json_serializable(obj):
    """
    Ensure all values in an object are JSON serializable by converting numpy types to native Python types.
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _create_story_timeline(articles, entity_name: str) -> List[TimelineEvent]:
    """
    Create a coherent story timeline by clustering related articles and ordering them intelligently.
    Uses TF-IDF vectorization and clustering to group related storylines.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    if not articles:
        return []
    
    # Prepare text data for analysis
    article_texts = []
    article_data = []
    
    for article in articles:
        # Combine title, summary, and content for text analysis
        title = article[1] or ""
        summary = article[3] or ""
        content = article[2] or ""
        
        # Create combined text (prioritize title and summary)
        combined_text = f"{title} {summary} {content[:500]}"  # Limit content to avoid overwhelming
        article_texts.append(combined_text)
        article_data.append(article)
    
    # Skip clustering if we have too few articles
    if len(articles) < 2:
        # Single article - create simple timeline
        article = articles[0]
        event = TimelineEvent(
            id=f"event_{article[0]}_0",
            title=article[1],
            description=article[3] or article[2][:200] + "...",
            date=article[7],
            event_type=EventType.NEWS,
            sentiment=SentimentType.NEUTRAL,
            relevance_score=_calculate_relevance(article, entity_name),
            entities=article[9] or [],
            source_article_id=str(article[0]),
            source_url=article[4],
            metadata={
                "source": article[5],
                "author": article[6],
                "categories": article[8] or [],
                "source_credibility": _calculate_source_credibility(article[5]),
                "story_cluster": 0,
                "story_relevance": 1.0
            }
        )
        return [event]
    
    try:
        # Create TF-IDF vectors with enhanced preprocessing
        vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased for better differentiation
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            min_df=1,
            max_df=0.8,
            sublinear_tf=True,  # Apply log scaling
            norm='l2'  # L2 normalization
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(article_texts)
            logger.info(f"Created TF-IDF matrix with shape: {tfidf_matrix.shape}")
            
            # Check for empty matrix or insufficient features
            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                logger.warning("Empty TF-IDF matrix, falling back to simple timeline")
                raise ValueError("Empty TF-IDF matrix")
                
            if tfidf_matrix.shape[1] < 2:
                logger.warning("Insufficient features for clustering, falling back to simple timeline")
                raise ValueError("Insufficient features for clustering")
                
        except Exception as vectorize_error:
            logger.error(f"TF-IDF vectorization failed: {vectorize_error}")
            raise vectorize_error
        
        # Use cosine similarity for clustering
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert similarity to distance matrix with proper bounds checking
        # Ensure values are in valid range [0, 1] and distance is non-negative
        similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
        distance_matrix = 1.0 - similarity_matrix
        
        # Ensure distance matrix is non-negative and symmetric
        distance_matrix = np.maximum(distance_matrix, 0.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Validate distance matrix
        if np.any(distance_matrix < 0):
            logger.error("Distance matrix contains negative values after correction")
            raise ValueError("Invalid distance matrix")
        
        logger.info(f"Distance matrix stats: min={np.min(distance_matrix):.4f}, max={np.max(distance_matrix):.4f}")
        
        # Perform clustering using DBSCAN
        # Use adaptive eps based on data size and more lenient min_samples
        data_size = len(articles)
        if data_size <= 5:
            eps = 0.4  # More lenient for small datasets
            min_samples = 2
        elif data_size <= 10:
            eps = 0.35
            min_samples = 2
        else:
            eps = 0.3  # Tighter clusters for larger datasets
            min_samples = 3
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        try:
            cluster_labels = clustering.fit_predict(distance_matrix)
            logger.info(f"DBSCAN clustering completed. Found {len(set(cluster_labels))} clusters (including noise)")
        except Exception as dbscan_error:
            logger.error(f"DBSCAN failed: {dbscan_error}")
            # Fallback to simple similarity-based grouping
            threshold = 0.7  # High similarity threshold
            cluster_labels = np.full(len(articles), -1)  # Start with all as noise
            
            # Simple grouping based on high similarity
            for i in range(len(articles)):
                if cluster_labels[i] == -1:  # Not yet assigned
                    cluster_id = max(cluster_labels) + 1 if len(set(cluster_labels)) > 1 else 0
                    cluster_labels[i] = cluster_id
                    
                    # Find similar articles
                    for j in range(i + 1, len(articles)):
                        if cluster_labels[j] == -1 and similarity_matrix[i][j] >= threshold:
                            cluster_labels[j] = cluster_id
            
            logger.info(f"Fallback clustering completed. Found {len(set(cluster_labels))} groups")
        
        # Group articles by cluster
        clustered_articles = {}
        for i, (article, label) in enumerate(zip(article_data, cluster_labels)):
            if label not in clustered_articles:
                clustered_articles[label] = []
            clustered_articles[label].append((article, i, similarity_matrix[i]))
        
        # Create timeline events from clustered articles
        events = []
        
        # Process each cluster to create story threads
        for cluster_id, cluster_articles in clustered_articles.items():
            # Sort articles in cluster by date
            cluster_articles.sort(key=lambda x: x[0][7])  # Sort by published_date
            
            # Calculate cluster relevance (average similarity to entity-related content)
            entity_relevance_scores = []
            for article, idx, similarity_scores in cluster_articles:
                relevance = _calculate_relevance(article, entity_name)
                entity_relevance_scores.append(relevance)
            
            avg_cluster_relevance = np.mean(entity_relevance_scores) if entity_relevance_scores else 0.0
            
            # Only include clusters with decent relevance to the entity
            # More lenient threshold for noise cluster (-1) which represents individual stories
            relevance_threshold = 0.2 if cluster_id == -1 else 0.3
            if avg_cluster_relevance < relevance_threshold:
                continue
            
            # Create events for this story cluster
            for i, (article, idx, similarity_scores) in enumerate(cluster_articles):
                # Calculate story relevance (how well this article fits the story thread)
                story_relevance = avg_cluster_relevance
                if i > 0:
                    # For follow-up articles, consider similarity to previous articles in cluster
                    prev_similarities = [
                        similarity_matrix[idx][prev_idx] 
                        for _, prev_idx, _ in cluster_articles[:i]
                    ]
                    prev_mean = float(np.mean(prev_similarities))
                    story_relevance = max(float(story_relevance), prev_mean)
                
                event = TimelineEvent(
                    id=f"event_{article[0]}_{len(events)}",
                    title=article[1],
                    description=article[3] or article[2][:200] + "...",
                    date=article[7],
                    event_type=EventType.NEWS,
                    sentiment=SentimentType.NEUTRAL,
                    relevance_score=_calculate_relevance(article, entity_name),
                    entities=article[9] or [],
                    source_article_id=str(article[0]),
                    source_url=article[4],
                    metadata={
                        "source": article[5],
                        "author": article[6],
                        "categories": article[8] or [],
                        "source_credibility": _calculate_source_credibility(article[5]),
                        "story_cluster": int(cluster_id) if cluster_id is not None else -1,
                        "story_relevance": float(story_relevance),
                        "cluster_position": int(i),
                        "cluster_size": int(len(cluster_articles))
                    }
                )
                events.append(event)
        
        # Sort final events by date to create chronological timeline
        events.sort(key=lambda x: x.date)
        
        # Limit to most relevant events if we have too many
        if len(events) > 20:
            # Sort by story relevance and keep top 20
            events.sort(key=lambda x: x.metadata.get("story_relevance", 0), reverse=True)
            events = events[:20]
            # Re-sort by date
            events.sort(key=lambda x: x.date)
        
        return events
        
    except Exception as e:
        logger.error(f"Error in story clustering: {e}")
        logger.info(f"Falling back to simple timeline for {len(articles)} articles")
        # Fallback to simple timeline if clustering fails
        events = []
        for article in articles:
            event = TimelineEvent(
                id=f"event_{article[0]}_{len(events)}",
                title=article[1],
                description=article[3] or article[2][:200] + "...",
                date=article[7],
                event_type=EventType.NEWS,
                sentiment=SentimentType.NEUTRAL,
                relevance_score=_calculate_relevance(article, entity_name),
                entities=article[9] or [],
                source_article_id=str(article[0]),
                source_url=article[4],
                metadata={
                    "source": article[5],
                    "author": article[6],
                    "categories": article[8] or [],
                    "source_credibility": _calculate_source_credibility(article[5]),
                    "story_cluster": -1,
                    "story_relevance": float(_calculate_relevance(article, entity_name))
                }
            )
            events.append(event)
        
        # Sort by date and limit
        events.sort(key=lambda x: x.date)
        return events[:15]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)