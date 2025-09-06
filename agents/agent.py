"""
AI Agent with LangGraph for news, knowledge graph queries, and flashcard generation.
"""
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Literal, TypedDict

from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from core.config import settings
from retrieval.fetch_news import fetch_gnews as fetch_gnews_api
from retrieval.knowledge_graph import KnowledgeGraph
from retrieval.vector_store import VectorStore

# Setup logging
logger = logging.getLogger(__name__)

# Initialize components
vs = VectorStore()
kg = KnowledgeGraph()

class ChatState(TypedDict):
    messages: List[BaseMessage]

# ============================================================================
# EXISTING TOOLS
# ============================================================================

@tool
def fetch_gnews(query: str) -> str:
    """
    Fetch latest news articles based on a search query.
    
    Args:
        query: The search term for news articles
        
    Returns:
        JSON string containing news articles with title, content, summary, url, source, etc.
    """
    try:
        results = fetch_gnews_api(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return json.dumps({"error": f"Failed to fetch news: {str(e)}"})

@tool
def search_articles(query: str, top_k: int = 5) -> str:
    """
    Search stored articles using semantic similarity.
    
    Args:
        query: User search text
        top_k: Number of top matches to return (default: 5)
        
    Returns:
        JSON string with article details from database.
    """
    try:
        embedding = vs.embedding_generator.generate_embeddings(query)
        results = vs.pinecone_index.query(
            vector=embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        articles = []
        with vs.pg_conn.cursor() as cursor:
            for match in results["matches"]:
                article_id = match["metadata"]["article_id"]
                cursor.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
                row = cursor.fetchone()
                
                if row:
                    articles.append({
                        "id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "summary": row[3],
                        "url": row[4],
                        "source": row[5],
                        "author": row[6],
                        "published_date": str(row[7]),
                        "categories": row[8],
                        "entities": row[9],
                        "relevance_score": row[10],
                        "similarity_score": match["score"]
                    })
        
        return json.dumps(articles, indent=2)
        
    except Exception as e:
        logger.error(f"Error searching articles: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}"})

@tool
def query_knowledge_graph(question: str) -> str:
    """
    Query the knowledge graph for entity relationships and information.
    
    Args:
        question: Natural language question about entities and relationships
        
    Returns:
        JSON string containing query results and analysis.
    """
    try:
        result = kg.analyze_question_and_query(question)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Knowledge graph query error: {e}")
        return json.dumps({"error": f"Knowledge graph query failed: {str(e)}"})

@tool
def search_entity_relationships(entity1: str, entity2: str) -> str:
    """
    Find direct relationships between two specific entities.
    
    Args:
        entity1: First entity name
        entity2: Second entity name
        
    Returns:
        JSON string containing relationships between the entities.
    """
    try:
        cypher_query = """
        MATCH (n)-[r]-(m) 
        WHERE (toLower(n.name) CONTAINS toLower($entity1) AND toLower(m.name) CONTAINS toLower($entity2)) 
           OR (toLower(n.name) CONTAINS toLower($entity2) AND toLower(m.name) CONTAINS toLower($entity1)) 
        RETURN n.name as source, type(r) as relationship, m.name as target, 
               labels(n) as source_labels, labels(m) as target_labels
        """
        
        results = kg.run_query(cypher_query, {"entity1": entity1, "entity2": entity2})
        
        return json.dumps({
            "success": True,
            "entity1": entity1,
            "entity2": entity2,
            "relationships_found": len(results),
            "relationships": results
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Entity relationship search error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Relationship search failed: {str(e)}"
        })


# ============================================================================
# ENHANCED FLASHCARD TOOLS
# ============================================================================

@tool
def generate_flashcards_from_content(
    topic: str, 
    source_type: str = "recent_news",
    difficulty: str = "medium", 
    count: int = 10
) -> str:
    """
    Generate educational flashcards from various content sources with enhanced news focus.
    
    Args:
        topic: The subject/topic for flashcard generation
        source_type: Source to generate from (recent_news, knowledge_base, entities)
        difficulty: Difficulty level (easy, medium, hard)
        count: Number of flashcards to generate
        
    Returns:
        JSON string with generated flashcards
    """
    try:
        flashcards = []
        
        if source_type == "recent_news":
            # Enhanced news flashcard generation
            news_result = fetch_gnews(topic)
            news_data = json.loads(news_result)
            
            if "articles" in news_data and news_data["articles"]:
                articles = news_data["articles"][:min(5, len(news_data["articles"]))]
                
                # Generate multiple types of questions per article
                for i, article in enumerate(articles):
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = article.get('content', description)
                    source_name = article.get('source', {}).get('name', 'News Source')
                    published_date = article.get('publishedAt', '')
                    
                    # Question type variations for news articles
                    question_templates = [
                        {
                            "type": "Breaking News",
                            "question": f"What recent development has been reported regarding {topic}?",
                            "answer": f"According to {source_name}: {description[:300]}{'...' if len(description) > 300 else ''}",
                            "context": f"Breaking news from {source_name}"
                        },
                        {
                            "type": "Current Events",
                            "question": f"What is the latest update on {topic} according to recent news?",
                            "answer": f"Recent reporting indicates: {content[:400] if content else description[:400]}{'...' if len(content or description) > 400 else ''}",
                            "context": f"Current events update from {source_name}"
                        },
                        {
                            "type": "News Analysis",
                            "question": f"Why is {topic} currently making headlines?",
                            "answer": f"This story is significant because: {description[:250]}{'...' if len(description) > 250 else ''}\n\nSource: {source_name}",
                            "context": f"News analysis from {published_date[:10] if published_date else 'recent'}"
                        },
                        {
                            "type": "Fact Check",
                            "question": f"What are the key facts about the recent {topic} news?",
                            "answer": f"Key facts from {source_name}:\n• {description[:200]}{'...' if len(description) > 200 else ''}\n• Published: {published_date[:10] if published_date else 'Recently'}",
                            "context": "Fact-based news summary"
                        }
                    ]
                    
                    # Create 2-3 flashcards per article with different question types
                    cards_per_article = min(2, count - len(flashcards))
                    for j in range(cards_per_article):
                        if len(flashcards) >= count:
                            break
                            
                        template = question_templates[j % len(question_templates)]
                        
                        flashcard = {
                            "id": f"news_{topic.replace(' ', '_')}_{i}_{j}_{uuid.uuid4().hex[:8]}",
                            "category": f"News - {topic.title()}",
                            "question": template["question"],
                            "answer": template["answer"],
                            "difficulty": difficulty,
                            "tags": ["current_events", "news", topic.lower().replace(" ", "_"), template["type"].lower().replace(" ", "_")],
                            "status": "new",
                            "questionType": template["type"],
                            "newsDate": published_date,
                            "source": source_name,
                            "context": template["context"],
                            "readMore": article.get('url', '')
                        }
                        flashcards.append(flashcard)
                        
                        if len(flashcards) >= count:
                            break
                    
                    if len(flashcards) >= count:
                        break
                        
        elif source_type == "entities":
            # Generate from knowledge graph entities
            kg_result = query_knowledge_graph(f"Find entities and relationships related to {topic}")
            kg_data = json.loads(kg_result)
            
            if kg_data.get("entities"):
                entities = kg_data["entities"][:count]
                for i, entity in enumerate(entities):
                    entity_name = entity.get('name', f'Entity {i+1}')
                    
                    question_templates = [
                        f"Who or what is {entity_name} in relation to {topic}?",
                        f"What role does {entity_name} play in {topic}?",
                        f"How is {entity_name} connected to {topic}?",
                        f"What should you know about {entity_name} regarding {topic}?"
                    ]
                    
                    flashcard = {
                        "id": f"entity_{topic.replace(' ', '_')}_{i}_{uuid.uuid4().hex[:8]}",
                        "category": f"Knowledge Graph - {topic}",
                        "question": question_templates[i % len(question_templates)],
                        "answer": f"{entity_name} is connected to {topic} through: {entity.get('relationship', 'various relationships')}. {entity.get('description', 'Additional context available in knowledge graph.')}",
                        "difficulty": difficulty,
                        "tags": ["knowledge_graph", topic.lower().replace(" ", "_"), "entities"],
                        "status": "new",
                        "questionType": "Entity Relationship"
                    }
                    flashcards.append(flashcard)
                    
        else:  # knowledge_base (search stored articles)
            kb_result = search_articles(topic, top_k=min(5, count))
            kb_data = json.loads(kb_result)
            
            if kb_data and not kb_data.get("error"):
                articles = kb_data[:min(3, len(kb_data))]
                
                for i, article in enumerate(articles):
                    content = article.get("content", article.get("summary", ""))
                    
                    question_templates = [
                        f"What are the key points about {topic}?",
                        f"How would you explain {topic} based on available information?",
                        f"What important details should you know about {topic}?",
                        f"What context is important for understanding {topic}?"
                    ]
                    
                    flashcard = {
                        "id": f"kb_{topic.replace(' ', '_')}_{i}_{uuid.uuid4().hex[:8]}",
                        "category": f"Knowledge Base - {topic}",
                        "question": question_templates[i % len(question_templates)],
                        "answer": content[:500] + "..." if len(content) > 500 else content,
                        "difficulty": difficulty,
                        "tags": ["knowledge_base", topic.lower().replace(" ", "_")],
                        "status": "new",
                        "source": article.get("source", "Knowledge Base"),
                        "questionType": "Knowledge Review"
                    }
                    flashcards.append(flashcard)
                    
                    if len(flashcards) >= count:
                        break
        
        # If we still need more cards, generate conceptual ones
        while len(flashcards) < count:
            conceptual_questions = [
                f"What are the main aspects of {topic} that are currently relevant?",
                f"Why is {topic} significant in current events?",
                f"What should someone know to stay informed about {topic}?",
                f"How does {topic} impact current affairs?",
                f"What are the key developments in {topic}?"
            ]
            
            q_index = len(flashcards) % len(conceptual_questions)
            
            flashcard = {
                "id": f"concept_{topic.replace(' ', '_')}_{len(flashcards)}_{uuid.uuid4().hex[:8]}",
                "category": f"Conceptual - {topic}",
                "question": conceptual_questions[q_index],
                "answer": f"This is a conceptual question about {topic}. The answer would be based on current information, recent developments, and key facts related to this topic.",
                "difficulty": difficulty,
                "tags": ["conceptual", topic.lower().replace(" ", "_")],
                "status": "new",
                "questionType": "Conceptual Understanding"
            }
            flashcards.append(flashcard)
        
        result = {
            "success": True,
            "flashcards": flashcards[:count],
            "source_type": source_type,
            "topic": topic,
            "generated_count": len(flashcards[:count])
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Enhanced flashcard generation error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "flashcards": [],
            "generated_count": 0
        })

@tool
def get_flashcards_for_study(
    category: str = None,
    difficulty: str = None, 
    status: str = None,
    due_for_review: bool = False,
    limit: int = 20
) -> str:
    """
    Retrieve flashcards for study sessions with enhanced filtering and smart selection.
    """
    try:
        store = VectorStore()
        query = """
            SELECT id, category, question, answer, difficulty, tags, status, 
                   created_at, last_reviewed, review_count, success_rate, next_review,
                   source_type, news_date, source_url, context
            FROM flashcards WHERE 1=1
        """
        params = []
        
        if category:
            query += " AND category ILIKE %s"
            params.append(f"%{category}%")
        if difficulty:
            query += " AND difficulty = %s"
            params.append(difficulty)
        if status:
            query += " AND status = %s"
            params.append(status)
        if due_for_review:
            query += " AND (next_review IS NULL OR next_review <= %s OR status = 'new')"
            params.append(datetime.now())
            
        # Smart ordering: prioritize new cards and due reviews
        query += """ ORDER BY 
                    CASE WHEN status = 'new' THEN 1
                         WHEN next_review IS NOT NULL AND next_review <= %s THEN 2
                         WHEN status = 'learning' THEN 3
                         ELSE 4 END,
                    CASE WHEN news_date IS NOT NULL THEN news_date ELSE created_at END DESC,
                    last_reviewed ASC NULLS FIRST
                    LIMIT %s
                """
        params.extend([datetime.now(), limit])
        
        with store.pg_conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            flashcards = []
            for row in rows:
                flashcards.append({
                    "id": row[0],
                    "category": row[1],
                    "question": row[2],
                    "answer": row[3],
                    "difficulty": row[4],
                    "tags": json.loads(row[5]) if row[5] else [],
                    "status": row[6],
                    "created_at": str(row[7]) if row[7] else None,
                    "last_reviewed": str(row[8]) if row[8] else None,
                    "review_count": row[9] or 0,
                    "success_rate": float(row[10]) if row[10] else 0.0,
                    "next_review": str(row[11]) if row[11] else None,
                    "source_type": row[12],
                    "news_date": str(row[13]) if row[13] else None,
                    "source_url": row[14],
                    "context": row[15]
                })
        
        store.close()
        
        return json.dumps({
            "success": True,
            "flashcards": flashcards,
            "count": len(flashcards),
            "filters_applied": {
                "category": category,
                "difficulty": difficulty,
                "status": status,
                "due_for_review": due_for_review
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Enhanced flashcard retrieval error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Retrieval failed: {str(e)}",
            "flashcards": []
        })

# Keep existing tools
@tool
def fetch_gnews(query: str) -> str:
    """Fetch latest news articles based on a search query."""
    try:
        results = fetch_gnews_api(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return json.dumps({"error": f"Failed to fetch news: {str(e)}"})

@tool
def search_articles(query: str, top_k: int = 5) -> str:
    """Search stored articles using semantic similarity."""
    try:
        embedding = vs.embedding_generator.generate_embeddings(query)
        results = vs.pinecone_index.query(
            vector=embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        articles = []
        with vs.pg_conn.cursor() as cursor:
            for match in results["matches"]:
                article_id = match["metadata"]["article_id"]
                cursor.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
                row = cursor.fetchone()
                
                if row:
                    articles.append({
                        "id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "summary": row[3],
                        "url": row[4],
                        "source": row[5],
                        "author": row[6],
                        "published_date": str(row[7]),
                        "similarity_score": match["score"]
                    })
        
        return json.dumps(articles, indent=2)
        
    except Exception as e:
        logger.error(f"Error searching articles: {e}")
        return json.dumps({"error": f"Search failed: {str(e)}"})

@tool
def query_knowledge_graph(question: str) -> str:
    """Query the knowledge graph for entity relationships and information."""
    try:
        result = kg.analyze_question_and_query(question)
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Knowledge graph query error: {e}")
        return json.dumps({"error": f"Knowledge graph query failed: {str(e)}"})

@tool
def record_flashcard_review(card_id: str, correct: bool, difficulty_rating: int = None) -> str:
    """Record flashcard review with enhanced spaced repetition algorithm."""
    try:
        store = VectorStore()
        with store.pg_conn.cursor() as cursor:
            # Get current flashcard data
            cursor.execute("""
                SELECT review_count, success_rate, status, difficulty 
                FROM flashcards WHERE id = %s
            """, (card_id,))
            row = cursor.fetchone()
            
            if not row:
                return json.dumps({
                    "success": False,
                    "error": "Flashcard not found"
                })
            
            current_review_count, current_success_rate, current_status, card_difficulty = row
            current_review_count = current_review_count or 0
            current_success_rate = float(current_success_rate) if current_success_rate else 0.0
            
            # Calculate new success rate
            new_success_rate = (
                (current_success_rate * current_review_count + (1.0 if correct else 0.0)) 
                / (current_review_count + 1)
            )
            
            # Enhanced spaced repetition algorithm
            if correct:
                if current_review_count == 0:
                    interval_days = 1
                elif current_review_count == 1:
                    interval_days = 3
                elif current_review_count == 2:
                    interval_days = 7
                else:
                    # Progressive intervals based on performance
                    base_interval = min(90, 14 * (1.3 ** (current_review_count - 3)))
                    
                    # Adjust for difficulty rating and card difficulty
                    difficulty_modifier = 1.0
                    if difficulty_rating:
                        modifiers = {1: 1.8, 2: 1.3, 3: 1.0, 4: 0.7, 5: 0.5}
                        difficulty_modifier = modifiers.get(difficulty_rating, 1.0)
                    
                    # Adjust for card inherent difficulty
                    card_modifier = {"easy": 1.2, "medium": 1.0, "hard": 0.8}.get(card_difficulty, 1.0)
                    
                    interval_days = max(1, int(base_interval * new_success_rate * difficulty_modifier * card_modifier))
            else:
                # Incorrect answer: shorter intervals based on previous performance
                if new_success_rate < 0.3:
                    interval_days = 1  # Review tomorrow
                elif new_success_rate < 0.6:
                    interval_days = 2  # Review in 2 days
                else:
                    interval_days = 3  # Review in 3 days
            
            next_review = datetime.now() + timedelta(days=interval_days)
            
            # Determine new status based on performance
            if new_success_rate >= 0.9 and current_review_count >= 3:
                new_status = "known"
            elif new_success_rate >= 0.7 and current_review_count >= 1:
                new_status = "learning"
            elif current_review_count >= 0:
                new_status = "learning" if correct else "new"
            else:
                new_status = "new"
            
            # Update flashcard
            cursor.execute("""
                UPDATE flashcards 
                SET last_reviewed = %s, 
                    review_count = review_count + 1,
                    success_rate = %s,
                    status = %s,
                    next_review = %s
                WHERE id = %s
            """, (datetime.now(), new_success_rate, new_status, next_review, card_id))
            
            # Record in study sessions table
            cursor.execute("""
                INSERT INTO study_sessions (session_id, card_id, correct, difficulty_rating, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", card_id, correct, difficulty_rating, datetime.now()))
            
            store.pg_conn.commit()
        
        store.close()
        
        return json.dumps({
            "success": True,
            "message": "Review recorded successfully",
            "card_id": card_id,
            "was_correct": correct,
            "new_status": new_status,
            "success_rate": round(new_success_rate, 3),
            "next_review": next_review.isoformat(),
            "days_until_next_review": interval_days,
            "total_reviews": current_review_count + 1
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Review recording error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Review recording failed: {str(e)}"
        })

@tool
def get_study_statistics() -> str:
    """Get comprehensive study statistics with news-focused analytics."""
    try:
        store = VectorStore()
        with store.pg_conn.cursor() as cursor:
            # Overall statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status = 'known' THEN 1 END) as known,
                    COUNT(CASE WHEN status = 'learning' THEN 1 END) as learning,
                    COUNT(CASE WHEN status = 'new' THEN 1 END) as new,
                    AVG(review_count) as avg_reviews,
                    AVG(success_rate) as avg_success_rate,
                    COUNT(CASE WHEN next_review IS NOT NULL AND next_review <= %s THEN 1 END) as due_today,
                    COUNT(CASE WHEN source_type = 'recent_news' THEN 1 END) as news_cards
                FROM flashcards
            """, (datetime.now(),))
            overall = cursor.fetchone()
            
            # News-specific statistics
            cursor.execute("""
                SELECT 
                    DATE(news_date) as news_day,
                    COUNT(*) as cards_count,
                    AVG(success_rate) as avg_performance
                FROM flashcards 
                WHERE news_date IS NOT NULL AND news_date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY DATE(news_date)
                ORDER BY news_day DESC
                LIMIT 10
            """)
            news_stats = cursor.fetchall()
            
            # Category breakdown
            cursor.execute("""
                SELECT category, 
                       COUNT(*) as total_count, 
                       COUNT(CASE WHEN status = 'known' THEN 1 END) as mastered_count,
                       AVG(success_rate) as avg_success_rate,
                       AVG(review_count) as avg_reviews
                FROM flashcards 
                GROUP BY category 
                HAVING COUNT(*) > 0
                ORDER BY total_count DESC
            """)
            categories = cursor.fetchall()
            
            # Study streak calculation
            cursor.execute("""
                SELECT COUNT(DISTINCT DATE(last_reviewed)) as study_days
                FROM flashcards 
                WHERE last_reviewed >= CURRENT_DATE - INTERVAL '30 days'
            """)
            study_streak = cursor.fetchone()[0] or 0
            
            # Recent performance trends
            cursor.execute("""
                SELECT 
                    DATE(created_at) as session_date,
                    COUNT(*) as cards_reviewed,
                    COUNT(CASE WHEN correct THEN 1 END) as correct_answers,
                    AVG(CASE WHEN difficulty_rating IS NOT NULL THEN difficulty_rating END) as avg_difficulty_rating
                FROM study_sessions 
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(created_at)
                ORDER BY session_date DESC
            """)
            recent_sessions = cursor.fetchall()
        
        store.close()
        
        # Calculate additional metrics
        mastery_rate = (overall[1] / overall[0] * 100) if overall[0] > 0 else 0
        avg_success = float(overall[5]) if overall[5] else 0
        news_card_percentage = (overall[7] / overall[0] * 100) if overall[0] > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if overall[6] > 0:
            recommendations.append(f"You have {overall[6]} cards due for review today!")
        
        if overall[3] > 15:
            recommendations.append(f"You have {overall[3]} new cards. Consider studying 10-15 cards per session for best retention.")
        
        if avg_success < 0.7:
            recommendations.append("Your success rate is below 70%. Try reviewing cards more frequently or focusing on difficult topics.")
        
        if news_card_percentage > 50:
            recommendations.append("Great focus on current events! Your news knowledge is up-to-date.")
        elif overall[7] == 0:
            recommendations.append("Consider adding some current events flashcards to stay informed about recent developments.")
        
        if study_streak == 0:
            recommendations.append("Start your study streak today! Consistent daily practice improves retention.")
        elif study_streak >= 7:
            recommendations.append(f"Excellent! You've studied {study_streak} days this month. Keep up the great habit!")
        
        result = {
            "success": True,
            "overall_stats": {
                "total_cards": overall[0],
                "known": overall[1],
                "learning": overall[2],
                "new": overall[3],
                "average_reviews": round(float(overall[4]) if overall[4] else 0, 2),
                "average_success_rate": round(avg_success, 3),
                "due_today": overall[6],
                "news_cards": overall[7],
                "mastery_rate": round(mastery_rate, 1),
                "news_card_percentage": round(news_card_percentage, 1)
            },
            "categories": [
                {
                    "category": cat[0],
                    "total": cat[1],
                    "mastered": cat[2],
                    "mastery_rate": round(cat[2] / cat[1] * 100, 1) if cat[1] > 0 else 0,
                    "avg_success_rate": round(float(cat[3]) if cat[3] else 0, 3),
                    "avg_reviews": round(float(cat[4]) if cat[4] else 0, 1)
                }
                for cat in categories
            ],
            "news_statistics": [
                {
                    "date": str(news[0]),
                    "cards_created": news[1],
                    "avg_performance": round(float(news[2]) if news[2] else 0, 3)
                }
                for news in news_stats
            ],
            "study_streak_days": study_streak,
            "recent_sessions": [
                {
                    "date": str(session[0]),
                    "cards_reviewed": session[1],
                    "correct_answers": session[2],
                    "accuracy": round(session[2] / session[1] * 100, 1) if session[1] > 0 else 0,
                    "avg_difficulty_rating": round(float(session[3]) if session[3] else 0, 1)
                }
                for session in recent_sessions
            ],
            "recommendations": recommendations
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Statistics retrieval failed: {str(e)}"
        })

# Updated tool registry
TOOLS = [
    fetch_gnews, 
    search_articles, 
    query_knowledge_graph, 
    generate_flashcards_from_content,
    get_flashcards_for_study,
    record_flashcard_review,
    get_study_statistics
]

# ============================================================================
# AGENT NODES
# ============================================================================

def create_llm():
    """Create and configure the LLM instance."""
    return ChatOpenAI(
        model=settings.AGENT_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        streaming=True,
        temperature=0.1
    ).bind_tools(TOOLS)

def agent_node(state: ChatState) -> ChatState:
    """Main agent node that processes messages and generates responses."""
    llm = create_llm()
    
    # Enhanced system prompt for news flashcards
    system_prompt = """
    You are NewsNeuron, an AI assistant specialized in creating educational flashcards from current news and events. 
    Your primary focus is helping users stay informed about current events through spaced repetition learning.

    Key capabilities:
    - Generate flashcards from breaking news and current events
    - Create different types of questions (factual, analytical, contextual)
    - Track learning progress with spaced repetition algorithms
    - Provide study analytics and recommendations
    - Focus on recent developments and their significance

    When generating flashcards:
    - Prioritize recent, relevant news content
    - Create varied question types to test different aspects of understanding
    - Include context and source information
    - Make questions engaging and educational
    - Ensure answers are accurate and informative

    Available tools:
    - fetch_gnews: Get latest news articles
    - generate_flashcards_from_content: Create flashcards from various sources
    - get_flashcards_for_study: Retrieve cards for study sessions
    - record_flashcard_review: Track study progress
    - get_study_statistics: Provide learning analytics
    """
    
    messages = state["messages"]
    if not messages or not any(msg.type == "system" for msg in messages):
        messages = [ChatPromptTemplate.from_template(system_prompt).format()] + messages
    
    try:
        response = llm.invoke(messages)
        return {"messages": state["messages"] + [response]}
    except Exception as e:
        logger.error(f"Agent node error: {e}")
        error_msg = AIMessage(content=f"I apologize, but I encountered an error: {str(e)}")
        return {"messages": state["messages"] + [error_msg]}

def tools_node(state: ChatState) -> ChatState:
    """Execute tool calls from the last message."""
    last_message = state["messages"][-1]
    
    if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return {"messages": state["messages"]}
    
    tool_map = {tool.name: tool for tool in TOOLS}
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        try:
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
            else:
                result = f"Error: Unknown tool '{tool_name}'"
                
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            result = f"Error executing {tool_name}: {str(e)}"
        
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    
    return {"messages": state["messages"] + tool_results}

def route_decision(state: ChatState) -> Literal["tools", "__end__"]:
    """Determine whether to continue with tools or end the conversation."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "__end__"

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph():
    """Create and compile the agent workflow graph."""
    workflow = StateGraph(ChatState)
    
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_decision, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# Create the compiled graph
agent_graph = create_agent_graph()