"""
FastAPI routes for the AI agent application.
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

from agents.agent import agent_graph
from retrieval.fetch_news import fetch_gnews
from retrieval.vector_store import VectorStore
from core.models import NewsArticle
from core.config import settings

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
    
    
from typing import Optional


class FlashcardRequest(BaseModel):
    topic: Optional[str] = None   # e.g. "climate change", "politics"
    source_type: str = "news"
    count: int = 10
    difficulty: str = "medium"

class ReviewRequest(BaseModel):
    correct: bool
    difficulty_rating: Optional[int] = None
@app.post("/flashcards/generate")
async def generate_flashcards(request: FlashcardRequest) -> Dict[str, Any]:
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
            raise HTTPException(status_code=500, detail="Flashcard generation tool not available")
        
        result = generate_tool.invoke({
            "topic": request.topic,
            "source_type": request.source_type,
            "difficulty": request.difficulty,
            "count": request.count
        })
        
        flashcards_data = json.loads(result)
        
        if not flashcards_data.get("success"):
            raise HTTPException(status_code=400, detail=flashcards_data.get("error", "Generation failed"))
        
        # Store flashcards in database
        store = VectorStore()
        saved_cards = []
        
        try:
            with store.pg_conn.cursor() as cursor:
                for card in flashcards_data.get("flashcards", []):
                    cursor.execute("""
                        INSERT INTO flashcards 
                        (id, category, question, answer, difficulty, tags, status, created_at, success_rate)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        RETURNING id
                    """, (
                        card["id"], card["category"], card["question"], card["answer"],
                        card["difficulty"], json.dumps(card["tags"]), card["status"],
                        datetime.now(), 0.0
                    ))
                    
                    if cursor.fetchone():  # Card was inserted
                        saved_cards.append(card)
                
                store.pg_conn.commit()
        finally:
            store.close()
        
        return {
            "success": True,
            "flashcards": flashcards_data.get("flashcards", []),
            "count": len(saved_cards),
            "message": f"Generated and saved {len(saved_cards)} flashcards for topic: {request.topic}"
        }
        
    except Exception as e:
        logger.error(f"Flashcard generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/flashcards")
async def get_flashcards(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    status: Optional[str] = None,
    due_for_review: bool = False,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Retrieve flashcards with advanced filters using the agent's tools.
    """
    try:
        from agents.agent import TOOLS
        
        get_tool = next(
            (tool for tool in TOOLS if tool.name == "get_flashcards_for_study"), 
            None
        )
        
        if not get_tool:
            raise HTTPException(status_code=500, detail="Flashcard retrieval tool not available")
        
        result = get_tool.invoke({
            "category": category,
            "difficulty": difficulty,
            "status": status,
            "due_for_review": due_for_review,
            "limit": limit
        })
        
        data = json.loads(result)
        
        if not data.get("success"):
            raise HTTPException(status_code=500, detail=data.get("error", "Retrieval failed"))
        
        return data
        
    except Exception as e:
        logger.error(f"Flashcard retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.put("/flashcards/{card_id}/review")
async def review_flashcard(card_id: str, review: ReviewRequest) -> Dict[str, Any]:
    """
    Record flashcard review result using the agent's spaced repetition tool.
    """
    try:
        from agents.agent import TOOLS
        
        review_tool = next(
            (tool for tool in TOOLS if tool.name == "record_flashcard_review"), 
            None
        )
        
        if not review_tool:
            raise HTTPException(status_code=500, detail="Review recording tool not available")
        
        result = review_tool.invoke({
            "card_id": card_id,
            "correct": review.correct,
            "difficulty_rating": review.difficulty_rating
        })
        
        data = json.loads(result)
        
        if not data.get("success"):
            if "not found" in data.get("error", "").lower():
                raise HTTPException(status_code=404, detail="Flashcard not found")
            raise HTTPException(status_code=500, detail=data.get("error", "Review recording failed"))
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Review recording error: {e}")
        raise HTTPException(status_code=500, detail=f"Review recording failed: {str(e)}")

@app.post("/flashcards/study-session")
async def start_study_session(topics: Optional[List[str]] = None, max_cards: int = 20) -> Dict[str, Any]:
    """
    Start a new study session with optimized card selection.
    """
    try:
        from agents.agent import TOOLS
        
        get_tool = next(
            (tool for tool in TOOLS if tool.name == "get_flashcards_for_study"), 
            None
        )
        
        if not get_tool:
            raise HTTPException(status_code=500, detail="Study session tool not available")
        
        # Get cards due for review or new cards
        result = get_tool.invoke({
            "due_for_review": True,
            "limit": max_cards
        })
        
        data = json.loads(result)
        
        if not data.get("success"):
            raise HTTPException(status_code=500, detail=data.get("error", "Session creation failed"))
        
        flashcards = data.get("flashcards", [])
        
        # Filter by topics if specified
        if topics:
            flashcards = [
                card for card in flashcards 
                if any(topic.lower() in card.get("category", "").lower() for topic in topics)
            ]
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_cards = []
        for card in flashcards[:max_cards]:
            session_cards.append({
                "id": card["id"],
                "question": card["question"],
                "answer": card["answer"],
                "difficulty": card["difficulty"],
                "category": card["category"]
            })
        
        return {
            "session_id": session_id,
            "cards": session_cards,
            "total_cards": len(session_cards),
            "start_time": datetime.now().isoformat(),
            "topics": topics or ["all"]
        }
        
    except Exception as e:
        logger.error(f"Study session error: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.get("/flashcards/stats")
async def get_study_stats() -> Dict[str, Any]:
    """
    Get comprehensive study statistics using the agent's analytics tool.
    """
    try:
        from agents.agent import TOOLS
        
        stats_tool = next(
            (tool for tool in TOOLS if tool.name == "get_study_statistics"), 
            None
        )
        
        if not stats_tool:
            raise HTTPException(status_code=500, detail="Statistics tool not available")
        
        result = stats_tool.invoke({})
        data = json.loads(result)
        
        if not data.get("success"):
            raise HTTPException(status_code=500, detail=data.get("error", "Statistics retrieval failed"))
        
        return data
        
    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

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
# DATABASE SETUP ENDPOINT
# ============================================================================

@app.post("/setup/flashcard-database")
async def setup_flashcard_database():
    """
    Set up the database schema for flashcards (run once).
    """
    try:
        store = VectorStore()
        
        with store.pg_conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS flashcards (
                    id VARCHAR(255) PRIMARY KEY,
                    category VARCHAR(255) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    difficulty VARCHAR(20) DEFAULT 'medium',
                    tags JSONB DEFAULT '[]',
                    status VARCHAR(20) DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_reviewed TIMESTAMP,
                    review_count INTEGER DEFAULT 0,
                    success_rate FLOAT DEFAULT 0.0,
                    next_review TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_flashcards_category ON flashcards(category);
                CREATE INDEX IF NOT EXISTS idx_flashcards_status ON flashcards(status);
                CREATE INDEX IF NOT EXISTS idx_flashcards_next_review ON flashcards(next_review);
                CREATE INDEX IF NOT EXISTS idx_flashcards_difficulty ON flashcards(difficulty);
            """)
            
            store.pg_conn.commit()
        
        store.close()
        
        return {
            "success": True,
            "message": "Flashcard database schema set up successfully"
        }
        
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Database setup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)