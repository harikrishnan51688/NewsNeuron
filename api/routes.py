# Enhanced NewsNeuron Features
# Updated version without user authentication, integrated with LangGraph

import asyncio
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from agents.agent import SYSTEM_PROMPT, app_graph
from core.models import NewsArticle, QueryRequest, RetrievalResult
from retrieval.fetch_news import fetch_gnews
from retrieval.vector_store import VectorStore

app = FastAPI()

# In-memory storage for simplicity (replace with Redis in production)
trending_cache = {}
article_cache = {}
global_preferences = {"default_interests": ["technology", "business", "politics"]}

# ---- Feature 1: Trending Topics Endpoint ----
@app.get("/trending")
def get_trending_topics():
    """Get currently trending news topics based on article frequency"""
    try:
        # Fetch recent articles from multiple general queries
        general_queries = ["breaking news", "technology", "politics", "business", "sports"]
        all_articles = []
        
        for query in general_queries:
            results = fetch_gnews(query, max_results=20)
            if 'articles' in results:
                all_articles.extend(results['articles'])
        
        # Count keyword frequency to find trending topics
        keyword_counts = defaultdict(int)
        for article in all_articles:
            # Simple keyword extraction from titles
            words = article.get('title', '').lower().split()
            for word in words:
                if len(word) > 4:  # Filter out short words
                    keyword_counts[word] += 1
        
        # Get top trending keywords
        trending = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        trending_cache['last_update'] = datetime.now()
        trending_cache['topics'] = [{"keyword": k, "count": v} for k, v in trending]
        
        return {"trending_topics": trending_cache['topics'], "last_updated": trending_cache['last_update']}
    
    except Exception as e:
        return {"error": str(e), "trending_topics": []}

# ---- Feature 2: Smart News Summarization with LangGraph ----
@app.post("/summarize")
async def summarize_articles(request: QueryRequest):
    """Get AI-powered summary of multiple articles on a topic using LangGraph"""
    try:
        # Fetch articles
        results = fetch_gnews(request.query, max_results=request.limit)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        # Prepare articles for summarization
        articles_text = ""
        for i, article in enumerate(results['articles'][:5]):  # Limit to 5 for processing
            articles_text += f"Article {i+1}: {article.get('title', '')}\n"
            articles_text += f"Summary: {article.get('description', '')}\n\n"
        
        # Use LangGraph agent to summarize
        messages = [
            SystemMessage(content="You are a news analyst. Provide a comprehensive summary of multiple articles on the same topic. Identify key themes, different perspectives, and important facts."),
            HumanMessage(content=f"Please analyze and summarize these articles about '{request.query}':\n\n{articles_text}")
        ]
        
        # Stream through LangGraph
        summary_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        summary_response = ai_msg.content
        
        return {
            "query": request.query,
            "summary": summary_response,
            "article_count": len(results['articles']),
            "sources": [article.get('source', {}).get('name', 'Unknown') for article in results['articles'][:5]]
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 3: Article Similarity & Related News ----
@app.get("/related/{article_title}")
def get_related_articles(article_title: str, limit: int = 5):
    """Find articles similar to a given article using keyword matching"""
    try:
        # Extract keywords from the article title
        keywords = [word for word in article_title.lower().split() if len(word) > 3]
        search_query = " ".join(keywords[:3])  # Use top 3 keywords
        
        # Search for related articles
        results = fetch_gnews(search_query, max_results=limit * 2)
        if not results.get('articles'):
            return {"related_articles": []}
        
        # Filter out the original article and get most relevant
        related = []
        for article in results['articles']:
            if article['title'] != article_title:
                related.append({
                    "title": article['title'],
                    "source": article['source']['name'],
                    "url": article['url'],
                    "published": article['publishedAt']
                })
                if len(related) >= limit:
                    break
        
        return {"related_articles": related}
    
    except Exception as e:
        return {"error": str(e), "related_articles": []}

# ---- Feature 4: News Timeline ----
@app.get("/timeline")
def get_news_timeline(topic: str, days: int = 7):
    """Get chronological timeline of news events for a topic"""
    try:
        # Fetch articles for the topic
        results = fetch_gnews(topic, max_results=50)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        # Group articles by date
        timeline = defaultdict(list)
        for article in results['articles']:
            try:
                pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                date_key = pub_date.strftime('%Y-%m-%d')
                timeline[date_key].append({
                    "title": article['title'],
                    "source": article['source']['name'],
                    "url": article['url'],
                    "time": pub_date.strftime('%H:%M')
                })
            except:
                continue
        
        # Sort timeline
        sorted_timeline = []
        for date in sorted(timeline.keys(), reverse=True):
            sorted_timeline.append({
                "date": date,
                "articles": sorted(timeline[date], key=lambda x: x['time'], reverse=True)
            })
        
        return {
            "topic": topic,
            "timeline": sorted_timeline[:days],
            "total_articles": len(results['articles'])
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 5: Global News Alerts (No User Auth) ----
@app.get("/alerts/global")
def get_global_alerts(keywords: Optional[str] = None):
    """Get global news alerts for popular topics"""
    try:
        # Default trending keywords if none provided
        if not keywords:
            alert_keywords = ["breaking news", "technology", "market crash", "election", "climate"]
        else:
            alert_keywords = [k.strip() for k in keywords.split(',')]
        
        alerts = []
        for keyword in alert_keywords:
            results = fetch_gnews(keyword, max_results=5)
            if results.get('articles'):
                # Get most recent article for this keyword
                recent_article = results['articles'][0]
                alerts.append({
                    "keyword": keyword,
                    "latest_article": {
                        "title": recent_article['title'],
                        "source": recent_article['source']['name'],
                        "url": recent_article['url'],
                        "published": recent_article['publishedAt']
                    },
                    "article_count": len(results['articles'])
                })
        
        return {
            "global_alerts": alerts,
            "last_updated": datetime.now()
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 6: News Sentiment Analysis with LangGraph ----
@app.post("/sentiment")
async def analyze_sentiment(request: QueryRequest):
    """Analyze sentiment of news articles on a topic using LangGraph"""
    try:
        results = fetch_gnews(request.query, max_results=request.limit)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        # Use LangGraph agent for sentiment analysis
        articles_for_analysis = []
        for article in results['articles'][:10]:
            articles_for_analysis.append(f"Title: {article['title']}\nSummary: {article.get('description', '')}")
        
        articles_text = "\n\n".join(articles_for_analysis)
        
        messages = [
            SystemMessage(content="You are a sentiment analyst. Analyze the overall sentiment of news articles and provide a brief analysis with sentiment scores (positive, negative, neutral percentages). Be objective and factual."),
            HumanMessage(content=f"Analyze the sentiment of these news articles about '{request.query}':\n\n{articles_text}")
        ]
        
        sentiment_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        sentiment_response = ai_msg.content
        
        return {
            "query": request.query,
            "sentiment_analysis": sentiment_response,
            "articles_analyzed": len(articles_for_analysis)
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 7: News Categories & Tagging with LangGraph ----
@app.post("/categorize")
async def categorize_article(article_text: str):
    """Automatically categorize a news article using LangGraph"""
    try:
        messages = [
            SystemMessage(content="You are a news categorization expert. Categorize the given article into one or more categories: Politics, Technology, Business, Sports, Entertainment, Health, Science, World, Local. Respond with just the categories as a comma-separated list."),
            HumanMessage(content=f"Categorize this article:\n\n{article_text}")
        ]
        
        category_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        category_response = ai_msg.content
        
        categories = [cat.strip() for cat in category_response.split(',')]
        
        return {"categories": categories}
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 8: News Fact-Checking with LangGraph ----
@app.post("/fact-check")
async def fact_check_claim(claim: str):
    """Fact-check a specific claim using multiple news sources and LangGraph"""
    try:
        # Search for articles related to the claim
        results = fetch_gnews(claim, max_results=10)
        if not results.get('articles'):
            return {"error": "No articles found for fact-checking"}
        
        # Prepare context for fact-checking
        context = ""
        sources = []
        for article in results['articles'][:5]:
            context += f"Source: {article['source']['name']}\n"
            context += f"Title: {article['title']}\n"
            context += f"Content: {article.get('description', '')}\n\n"
            sources.append(article['source']['name'])
        
        messages = [
            SystemMessage(content="You are a fact-checker. Analyze the claim against the provided news sources. Provide a fact-check verdict (True/False/Partially True/Insufficient Evidence) and a brief explanation. Be objective and cite specific sources."),
            HumanMessage(content=f"Fact-check this claim: '{claim}'\n\nBased on these news sources:\n{context}")
        ]
        
        fact_check_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        fact_check_response = ai_msg.content
        
        return {
            "claim": claim,
            "fact_check_result": fact_check_response,
            "sources_checked": sources,
            "article_count": len(results['articles'])
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 9: Default News Feed (No User Auth) ----
@app.get("/feed")
def get_default_feed(limit: int = 20, interests: Optional[str] = None):
    """Get default news feed based on popular topics or provided interests"""
    try:
        # Use provided interests or default ones
        if interests:
            interest_list = [i.strip() for i in interests.split(',')]
        else:
            interest_list = global_preferences['default_interests']
        
        all_articles = []
        for interest in interest_list:
            results = fetch_gnews(interest, max_results=10)
            if results.get('articles'):
                for article in results['articles']:
                    article['category'] = interest
                    all_articles.append(article)
        
        # Remove duplicates and sort by relevance/date
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        # Sort by publication date (most recent first)
        unique_articles.sort(key=lambda x: x['publishedAt'], reverse=True)
        
        return {
            "news_feed": unique_articles[:limit],
            "interests": interest_list,
            "total_sources": len(set(a['source']['name'] for a in unique_articles))
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 10: News Comparison with LangGraph ----
@app.post("/compare")
async def compare_news_coverage(topic: str, sources: Optional[List[str]] = None):
    """Compare how different news sources cover the same topic using LangGraph"""
    try:
        results = fetch_gnews(topic, max_results=30)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        # Group articles by source
        source_coverage = defaultdict(list)
        for article in results['articles']:
            source_name = article['source']['name']
            if not sources or source_name in sources:
                source_coverage[source_name].append({
                    "title": article['title'],
                    "description": article.get('description', ''),
                    "url": article['url'],
                    "published": article['publishedAt']
                })
        
        # Use LangGraph agent to analyze coverage differences
        coverage_text = ""
        for source, articles in source_coverage.items():
            coverage_text += f"\n{source}:\n"
            for article in articles[:3]:  # Limit per source
                coverage_text += f"- {article['title']}\n"
        
        messages = [
            SystemMessage(content="You are a media analyst. Compare how different news sources cover the same topic. Identify different angles, biases, or perspectives. Be objective and factual."),
            HumanMessage(content=f"Compare news coverage of '{topic}' across different sources:\n{coverage_text}")
        ]
        
        analysis_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        analysis_response = ai_msg.content
        
        return {
            "topic": topic,
            "source_comparison": analysis_response,
            "sources_analyzed": list(source_coverage.keys()),
            "total_articles": len(results['articles'])
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 11: News Analytics Dashboard Data ----
@app.get("/analytics")
def get_news_analytics(topic: Optional[str] = None, days: int = 7):
    """Get analytics data for news dashboard"""
    try:
        if topic:
            query_terms = [topic]
        else:
            query_terms = ["breaking news", "technology", "politics", "business"]
        
        analytics_data = {
            "total_articles": 0,
            "sources_breakdown": defaultdict(int),
            "daily_counts": defaultdict(int),
            "top_keywords": defaultdict(int)
        }
        
        for query in query_terms:
            results = fetch_gnews(query, max_results=50)
            if results.get('articles'):
                for article in results['articles']:
                    analytics_data["total_articles"] += 1
                    
                    # Source breakdown
                    source = article['source']['name']
                    analytics_data["sources_breakdown"][source] += 1
                    
                    # Daily counts
                    try:
                        pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                        date_key = pub_date.strftime('%Y-%m-%d')
                        analytics_data["daily_counts"][date_key] += 1
                    except:
                        pass
                    
                    # Keyword frequency
                    words = article['title'].lower().split()
                    for word in words:
                        if len(word) > 4:
                            analytics_data["top_keywords"][word] += 1
        
        # Convert to lists for JSON response
        return {
            "total_articles": analytics_data["total_articles"],
            "sources_breakdown": dict(analytics_data["sources_breakdown"]),
            "daily_counts": dict(analytics_data["daily_counts"]),
            "top_keywords": dict(sorted(analytics_data["top_keywords"].items(), 
                                      key=lambda x: x[1], reverse=True)[:20])
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 12: Breaking News Alerts ----
@app.get("/breaking")
def get_breaking_news():
    """Get latest breaking news with high priority"""
    try:
        # Fetch recent breaking news
        results = fetch_gnews("breaking news", max_results=10)
        if not results.get('articles'):
            return {"breaking_news": []}
        
        # Filter for very recent articles (last 6 hours)
        recent_cutoff = datetime.now() - timedelta(hours=6)
        breaking_articles = []
        
        for article in results['articles']:
            try:
                pub_date = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                if pub_date >= recent_cutoff:
                    breaking_articles.append({
                        "title": article['title'],
                        "source": article['source']['name'],
                        "url": article['url'],
                        "published": article['publishedAt'],
                        "urgency": "high" if pub_date >= datetime.now() - timedelta(hours=2) else "medium"
                    })
            except:
                continue
        
        return {
            "breaking_news": breaking_articles,
            "last_updated": datetime.now(),
            "count": len(breaking_articles)
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 13: Enhanced Search with Filters ----
@app.post("/search/advanced")
def advanced_search(request: QueryRequest):
    """Advanced search with filters and sorting"""
    try:
        # Fetch articles
        results = fetch_gnews(request.query, max_results=50)
        if not results.get('articles'):
            return {"articles": [], "total_count": 0}
        
        articles = results['articles']
        filters = getattr(request, 'filters', {}) or {}
        
        # Apply filters
        if filters.get('source'):
            articles = [a for a in articles if a['source']['name'] == filters['source']]
        
        if filters.get('date_from'):
            date_from = datetime.fromisoformat(filters['date_from'])
            articles = [a for a in articles 
                       if datetime.fromisoformat(a['publishedAt'].replace('Z', '+00:00')) >= date_from]
        
        if filters.get('date_to'):
            date_to = datetime.fromisoformat(filters['date_to'])
            articles = [a for a in articles 
                       if datetime.fromisoformat(a['publishedAt'].replace('Z', '+00:00')) <= date_to]
        
        # Sort articles
        sort_by = filters.get('sort_by', 'date')
        if sort_by == 'date':
            articles.sort(key=lambda x: x['publishedAt'], reverse=True)
        elif sort_by == 'relevance':
            # Simple relevance scoring based on query term frequency
            def relevance_score(article):
                title_lower = article['title'].lower()
                desc_lower = article.get('description', '').lower()
                query_lower = request.query.lower()
                return title_lower.count(query_lower) + desc_lower.count(query_lower)
            
            articles.sort(key=relevance_score, reverse=True)
        
        # Limit results
        limited_articles = articles[:request.limit]
        
        return {
            "articles": limited_articles,
            "total_count": len(articles),
            "applied_filters": filters,
            "query": request.query
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 14: News Export ----
@app.get("/export")
def export_news_data(query: str, format: str = "json"):
    """Export news data in different formats"""
    try:
        results = fetch_gnews(query, max_results=100)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        if format.lower() == "csv":
            # Simple CSV export
            csv_content = "Title,Source,URL,Published,Description\n"
            for article in results['articles']:
                # Escape quotes for CSV
                title = article["title"].replace('"', '""')
                source = article["source"]["name"].replace('"', '""')
                desc = article.get("description", "").replace('"', '""')
                csv_content += f'"{title}","{source}","{article["url"]}","{article["publishedAt"]}","{desc}"\n'
            
            return {
                "format": "csv",
                "content": csv_content,
                "article_count": len(results['articles'])
            }
        
        return {
            "format": "json",
            "content": results,
            "article_count": len(results['articles'])
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 15: Enhanced Chat with News Context ----
@app.post("/chat/news")
async def chat_with_news_context(query: str, news_topic: Optional[str] = None):
    """Chat with AI that has access to current news context via LangGraph"""
    try:
        # If news topic provided, fetch relevant news first
        news_context = ""
        if news_topic:
            results = fetch_gnews(news_topic, max_results=5)
            if results.get('articles'):
                news_context = "\n".join([
                    f"- {article['title']} ({article['source']['name']})"
                    for article in results['articles']
                ])
        
        # Prepare messages for LangGraph
        if news_context:
            system_msg = f"{SYSTEM_PROMPT}\n\nYou have access to current news about '{news_topic}':\n{news_context}\n\nUse this context to provide informed responses."
        else:
            system_msg = SYSTEM_PROMPT
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=query)
        ]
        
        # Stream response from LangGraph
        response_content = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        response_content = ai_msg.content
        
        return {
            "query": query,
            "response": response_content,
            "news_context_used": bool(news_context),
            "news_topic": news_topic
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 16: News Source Analysis ----
@app.get("/sources/analysis")
def analyze_news_sources(topic: str):
    """Analyze which sources are covering a topic and their coverage patterns"""
    try:
        results = fetch_gnews(topic, max_results=50)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        source_analysis = defaultdict(lambda: {
            "article_count": 0,
            "latest_article": None,
            "coverage_frequency": []
        })
        
        for article in results['articles']:
            source_name = article['source']['name']
            source_analysis[source_name]["article_count"] += 1
            
            # Track latest article
            if not source_analysis[source_name]["latest_article"]:
                source_analysis[source_name]["latest_article"] = {
                    "title": article['title'],
                    "published": article['publishedAt'],
                    "url": article['url']
                }
        
        # Convert to regular dict and sort by coverage
        sorted_sources = sorted(
            [(k, v) for k, v in source_analysis.items()],
            key=lambda x: x[1]["article_count"],
            reverse=True
        )
        
        return {
            "topic": topic,
            "source_analysis": dict(sorted_sources),
            "total_sources": len(source_analysis),
            "most_active_source": sorted_sources[0][0] if sorted_sources else None
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 17: News Insights with LangGraph ----
@app.post("/insights")
async def generate_news_insights(topic: str):
    """Generate insights and analysis about a news topic using LangGraph"""
    try:
        # Fetch comprehensive data about the topic
        results = fetch_gnews(topic, max_results=20)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        # Prepare data for analysis
        articles_summary = ""
        sources = set()
        for article in results['articles']:
            articles_summary += f"• {article['title']} - {article['source']['name']}\n"
            sources.add(article['source']['name'])
        
        messages = [
            SystemMessage(content="You are a news analyst. Provide insights about a news topic including: key developments, major players, implications, and trends. Be analytical and objective."),
            HumanMessage(content=f"Provide insights about the topic '{topic}' based on these recent articles:\n\n{articles_summary}")
        ]
        
        insights_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        insights_response = ai_msg.content
        
        return {
            "topic": topic,
            "insights": insights_response,
            "articles_analyzed": len(results['articles']),
            "sources_covered": list(sources)
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 18: Health Check & Stats ----
@app.get("/health")
def health_check():
    """Application health check and basic statistics"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "features_available": [
            "search", "chat", "trending", "summarize", "related", 
            "timeline", "alerts", "sentiment", "compare", "analytics",
            "breaking", "advanced_search", "export", "categorize",
            "feed", "sources", "insights", "chat_news"
        ],
        "cache_stats": {
            "trending_cache_size": len(trending_cache),
            "article_cache_size": len(article_cache)
        },
        "langgraph_integration": "active"
    }

# ---- Feature 19: Streaming News Updates ----
@app.get("/stream/breaking")
async def stream_breaking_news():
    """Stream breaking news updates"""
    async def news_generator():
        while True:
            try:
                breaking_data = get_breaking_news()
                yield f"data: {json.dumps(breaking_data)}\n\n"
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(news_generator(), media_type="text/plain")

# ---- Feature 20: Topic Deep Dive with LangGraph ----
@app.post("/deep-dive")
async def topic_deep_dive(topic: str):
    """Get comprehensive deep dive analysis of a topic using LangGraph"""
    try:
        # Fetch multiple types of information
        general_results = fetch_gnews(topic, max_results=15)
        breaking_results = fetch_gnews(f"breaking {topic}", max_results=10)
        analysis_results = fetch_gnews(f"{topic} analysis", max_results=10)
        
        all_articles = []
        if general_results.get('articles'):
            all_articles.extend(general_results['articles'])
        if breaking_results.get('articles'):
            all_articles.extend(breaking_results['articles'])
        if analysis_results.get('articles'):
            all_articles.extend(analysis_results['articles'])
        
        if not all_articles:
            return {"error": "No articles found for deep dive"}
        
        # Remove duplicates
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article['url'] not in seen_urls:
                seen_urls.add(article['url'])
                unique_articles.append(article)
        
        # Prepare comprehensive context for deep dive
        comprehensive_context = ""
        for i, article in enumerate(unique_articles[:10]):
            comprehensive_context += f"Article {i+1}:\n"
            comprehensive_context += f"Title: {article['title']}\n"
            comprehensive_context += f"Source: {article['source']['name']}\n"
            comprehensive_context += f"Summary: {article.get('description', '')}\n"
            comprehensive_context += f"Published: {article['publishedAt']}\n\n"
        
        messages = [
            SystemMessage(content="You are an expert news analyst. Provide a comprehensive deep-dive analysis covering: background context, key developments, major stakeholders, implications, different perspectives, and future outlook. Be thorough and analytical."),
            HumanMessage(content=f"Provide a comprehensive deep-dive analysis of '{topic}' based on this news coverage:\n\n{comprehensive_context}")
        ]
        
        deep_dive_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        deep_dive_response = ai_msg.content
        
        return {
            "topic": topic,
            "deep_dive_analysis": deep_dive_response,
            "total_articles_analyzed": len(unique_articles),
            "sources": list(set(a['source']['name'] for a in unique_articles)),
            "analysis_type": "comprehensive"
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 21: News Question Answering with LangGraph ----
@app.post("/ask")
async def ask_about_news(question: str, context_topic: Optional[str] = None):
    """Ask specific questions about news topics using LangGraph with news context"""
    try:
        # Fetch relevant news context if topic provided
        news_context = ""
        if context_topic:
            results = fetch_gnews(context_topic, max_results=10)
            if results.get('articles'):
                news_context = "\n".join([
                    f"• {article['title']} - {article['source']['name']}: {article.get('description', '')}"
                    for article in results['articles'][:5]
                ])
        
        # If no specific topic, try to extract topic from question for context
        if not news_context:
            # Simple keyword extraction for context
            question_words = question.lower().split()
            potential_topics = [word for word in question_words if len(word) > 4]
            if potential_topics:
                topic_query = " ".join(potential_topics[:2])
                results = fetch_gnews(topic_query, max_results=5)
                if results.get('articles'):
                    news_context = "\n".join([
                        f"• {article['title']} - {article['source']['name']}"
                        for article in results['articles'][:3]
                    ])
        
        # Prepare messages for LangGraph
        system_content = "You are a knowledgeable news assistant. Answer questions using the provided news context when available. Be accurate and cite sources when possible."
        if news_context:
            system_content += f"\n\nCurrent news context:\n{news_context}"
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=question)
        ]
        
        answer_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        answer_response = ai_msg.content
        
        return {
            "question": question,
            "answer": answer_response,
            "context_used": bool(news_context),
            "context_topic": context_topic
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Feature 22: News Impact Analysis with LangGraph ----
@app.post("/impact")
async def analyze_news_impact(topic: str):
    """Analyze the potential impact and implications of a news topic"""
    try:
        results = fetch_gnews(topic, max_results=15)
        if not results.get('articles'):
            return {"error": "No articles found"}
        
        # Prepare articles for impact analysis
        articles_data = ""
        for article in results['articles'][:8]:
            articles_data += f"Title: {article['title']}\n"
            articles_data += f"Source: {article['source']['name']}\n"
            articles_data += f"Summary: {article.get('description', '')}\n\n"
        
        messages = [
            SystemMessage(content="You are an impact analyst. Analyze news topics for their potential economic, social, political, and technological impacts. Provide a structured analysis with short-term and long-term implications."),
            HumanMessage(content=f"Analyze the impact and implications of '{topic}' based on this news coverage:\n\n{articles_data}")
        ]
        
        impact_response = ""
        async for chunk in app_graph.astream({"messages": messages}):
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        impact_response = ai_msg.content
        
        return {
            "topic": topic,
            "impact_analysis": impact_response,
            "articles_analyzed": len(results['articles'][:8]),
            "analysis_timestamp": datetime.now()
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---- Updated Global Preferences Management ----
@app.post("/preferences/global")
def update_global_preferences(interests: List[str], default_sources: Optional[List[str]] = None):
    """Update global default preferences for the application"""
    global_preferences.update({
        "default_interests": interests,
        "default_sources": default_sources or [],
        "updated_at": datetime.now()
    })
    
    return {
        "message": "Global preferences updated successfully",
        "preferences": global_preferences
    }

@app.get("/preferences/global")
def get_global_preferences():
    """Get current global preferences"""
    return global_preferences

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)