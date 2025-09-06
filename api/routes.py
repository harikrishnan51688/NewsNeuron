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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)