"""
FastAPI routes for the AI agent application.
"""
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

from agents.agent import agent_graph
from retrieval.fetch_news import fetch_gnews
from retrieval.vector_store import VectorStore
from core.models import (
    NewsArticle, TimelineRequest, TimelineResponse, TimelineEvent, 
    EventType, SentimentType, TimelineSummary, EntityMention
)
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
        
        for article in articles:
            # For now, all articles are treated as NEWS events
            event_type = EventType.NEWS
                
            event = TimelineEvent(
                id=f"event_{article[0]}_{len(events)}",
                title=article[1],
                description=article[3] or article[2][:200] + "...",
                date=article[7],
                event_type=event_type,
                sentiment=SentimentType.NEUTRAL,  # Remove sentiment analysis
                relevance_score=_calculate_relevance(article, request.entity_name),
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
            events.append(event)
            
            # Track entity mentions
            for entity in (article[9] or []):
                if entity.lower() != request.entity_name.lower():
                    if entity not in entity_mentions:
                        entity_mentions[entity] = {
                            "count": 0,
                            "relevance_sum": 0,
                            "first_seen": article[7],
                            "last_seen": article[7]
                        }
                    entity_mentions[entity]["count"] += 1
                    entity_mentions[entity]["relevance_sum"] += event.relevance_score
                    entity_mentions[entity]["last_seen"] = max(
                        entity_mentions[entity]["last_seen"], article[7]
                    )
        
        # Calculate statistics
        statistics = {
            "total_events": len(events),
            "articles_analyzed": len(articles),
            "time_span_days": (end_date - start_date).days,
            "average_relevance": sum(e.relevance_score for e in events) / len(events) if events else 0,
            "fresh_data_fetched": existing_count < 3  # Indicate if we fetched fresh data
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)