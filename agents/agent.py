"""
AI Agent with LangGraph for news, knowledge graph queries, and flashcard generation.
"""
import json
import logging
import uuid
from typing import List, TypedDict, Literal
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from core.config import settings
from retrieval.fetch_news import fetch_gnews as fetch_gnews_api
from retrieval.vector_store import VectorStore
from retrieval.knowledge_graph import KnowledgeGraph

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
# FLASHCARD TOOLS
# ============================================================================

@tool
def generate_flashcards_from_content(
    topic: str, 
    source_type: str = "knowledge_base",
    difficulty: str = "medium", 
    count: int = 5
) -> str:
    """
    Generate educational flashcards from various content sources.
    
    Args:
        topic: The subject/topic for flashcard generation
        source_type: Source to generate from (knowledge_base, recent_news, entities)
        difficulty: Difficulty level (easy, medium, hard)
        count: Number of flashcards to generate
        
    Returns:
        JSON string with generated flashcards
    """
    try:
        flashcards = []
        
        if source_type == "recent_news":
            # Generate from recent news articles
            news_result = fetch_gnews(topic)
            news_data = json.loads(news_result)
            
            if "articles" in news_data and news_data["articles"]:
                articles = news_data["articles"][:min(3, count)]  # Use top 3 articles or less
                
                for i, article in enumerate(articles):
                    # Create different types of questions based on news content
                    question_types = [
                        f"What recent development occurred regarding {topic}?",
                        f"According to recent news, what is happening with {topic}?",
                        f"What key information should you know about {topic} based on current events?"
                    ]
                    
                    flashcard = {
                        "id": f"news_{topic.replace(' ', '_')}_{i+1}_{uuid.uuid4().hex[:8]}",
                        "category": f"Current Events - {topic}",
                        "question": question_types[i % len(question_types)],
                        "answer": f"Recent news from {article.get('source', {}).get('name', 'news source')}: {article.get('description', article.get('content', 'No description available'))[:300]}...",
                        "difficulty": difficulty,
                        "tags": ["current_events", topic.lower().replace(" ", "_"), "news"],
                        "status": "new"
                    }
                    flashcards.append(flashcard)
                    
        elif source_type == "entities":
            # Generate from knowledge graph entities
            kg_result = query_knowledge_graph(f"Find entities and relationships related to {topic}")
            kg_data = json.loads(kg_result)
            
            if "entities" in kg_data and kg_data["entities"]:
                entities = kg_data["entities"][:count]
                for i, entity in enumerate(entities):
                    question_types = [
                        f"What is the role of {entity.get('name', 'this entity')} in {topic}?",
                        f"How does {entity.get('name', 'this entity')} relate to {topic}?",
                        f"What should you know about {entity.get('name', 'this entity')} in the context of {topic}?"
                    ]
                    
                    flashcard = {
                        "id": f"entity_{topic.replace(' ', '_')}_{i+1}_{uuid.uuid4().hex[:8]}",
                        "category": f"Knowledge Graph - {topic}",
                        "question": question_types[i % len(question_types)],
                        "answer": f"This entity has {entity.get('relationship', 'a connection')} with {topic}. {entity.get('description', 'Additional context available in knowledge graph.')}",
                        "difficulty": difficulty,
                        "tags": ["knowledge_graph", topic.lower().replace(" ", "_"), "entities"],
                        "status": "new"
                    }
                    flashcards.append(flashcard)
        
        else:  # knowledge_base (search stored articles)
            # Generate from stored knowledge base articles
            kb_result = search_articles(topic, top_k=3)
            kb_data = json.loads(kb_result)
            
            if kb_data and not kb_data.get("error"):
                articles = kb_data[:min(3, count)]
                
                for i, article in enumerate(articles):
                    question_types = [
                        f"What is the main concept of {topic}?",
                        f"What are the key characteristics of {topic}?",
                        f"What important information should you know about {topic}?",
                        f"How would you explain {topic} to someone?",
                        f"What are the main points about {topic}?"
                    ]
                    
                    # Use article content to create more specific answers
                    content = article.get("content", article.get("summary", ""))[:400]
                    
                    flashcard = {
                        "id": f"kb_{topic.replace(' ', '_')}_{i+1}_{uuid.uuid4().hex[:8]}",
                        "category": f"Knowledge Base - {topic}",
                        "question": question_types[i % len(question_types)],
                        "answer": content + "..." if content else f"Based on stored knowledge: Key information about {topic} from our database.",
                        "difficulty": difficulty,
                        "tags": ["knowledge_base", topic.lower().replace(" ", "_")],
                        "status": "new"
                    }
                    flashcards.append(flashcard)
        
        # Fill remaining slots with general questions if needed
        while len(flashcards) < count:
            general_questions = [
                f"What are some important facts about {topic}?",
                f"Why is {topic} significant or important?",
                f"What should someone know about {topic}?",
                f"How is {topic} relevant in today's context?",
                f"What are the main aspects of {topic}?"
            ]
            
            question_index = len(flashcards) % len(general_questions)
            
            flashcard = {
                "id": f"general_{topic.replace(' ', '_')}_{len(flashcards)+1}_{uuid.uuid4().hex[:8]}",
                "category": f"General Knowledge - {topic}",
                "question": general_questions[question_index],
                "answer": f"General knowledge about {topic} - this would be filled with specific information based on available sources and context.",
                "difficulty": difficulty,
                "tags": ["general", topic.lower().replace(" ", "_")],
                "status": "new"
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
        logger.error(f"Flashcard generation error: {e}")
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
    Retrieve flashcards for study sessions with filtering options.
    
    Args:
        category: Filter by category (optional)
        difficulty: Filter by difficulty level (easy, medium, hard)
        status: Filter by status (new, learning, known)
        due_for_review: Only get cards due for review
        limit: Maximum number of cards to return
        
    Returns:
        JSON string with flashcards ready for study
    """
    try:
        store = VectorStore()
        query = """
            SELECT id, category, question, answer, difficulty, tags, status, 
                   created_at, last_reviewed, review_count, success_rate, next_review
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
            query += " AND (next_review IS NULL OR next_review <= %s)"
            params.append(datetime.now())
            
        # Prioritize cards that need review
        query += """ ORDER BY 
                    CASE WHEN status = 'new' THEN 1
                         WHEN status = 'learning' THEN 2
                         ELSE 3 END,
                    last_reviewed ASC NULLS FIRST
                    LIMIT %s
                """
        params.append(limit)
        
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
                    "next_review": str(row[11]) if row[11] else None
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
        logger.error(f"Flashcard retrieval error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Retrieval failed: {str(e)}",
            "flashcards": []
        })

@tool
def record_flashcard_review(card_id: str, correct: bool, difficulty_rating: int = None) -> str:
    """
    Record the result of reviewing a flashcard and update its status using spaced repetition.
    
    Args:
        card_id: Unique identifier of the flashcard
        correct: Whether the answer was correct
        difficulty_rating: Optional difficulty rating from 1-5 (1=very easy, 5=very hard)
        
    Returns:
        JSON string with review results and next review date
    """
    try:
        store = VectorStore()
        with store.pg_conn.cursor() as cursor:
            # Get current flashcard data
            cursor.execute("""
                SELECT review_count, success_rate, status 
                FROM flashcards WHERE id = %s
            """, (card_id,))
            row = cursor.fetchone()
            
            if not row:
                return json.dumps({
                    "success": False,
                    "error": "Flashcard not found"
                })
            
            current_review_count, current_success_rate, current_status = row
            current_review_count = current_review_count or 0
            current_success_rate = float(current_success_rate) if current_success_rate else 0.0
            
            # Calculate new success rate
            new_success_rate = (
                (current_success_rate * current_review_count + (1.0 if correct else 0.0)) 
                / (current_review_count + 1)
            )
            
            # Spaced repetition algorithm
            if correct:
                if current_review_count == 0:
                    interval_days = 1
                elif current_review_count == 1:
                    interval_days = 3
                else:
                    # Adjust based on success rate and difficulty rating
                    base_interval = 7 * (1.3 ** (current_review_count - 2))
                    difficulty_modifier = 1.0
                    
                    if difficulty_rating:
                        # 1=very easy (1.5x), 2=easy (1.2x), 3=normal (1.0x), 4=hard (0.8x), 5=very hard (0.6x)
                        modifiers = {1: 1.5, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.6}
                        difficulty_modifier = modifiers.get(difficulty_rating, 1.0)
                    
                    interval_days = min(90, int(base_interval * new_success_rate * difficulty_modifier))
            else:
                interval_days = 1  # Review again tomorrow if incorrect
            
            next_review = datetime.now() + timedelta(days=interval_days)
            
            # Determine new status
            if new_success_rate >= 0.85 and current_review_count >= 2:
                new_status = "known"
            elif current_review_count >= 0:
                new_status = "learning"
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
            "days_until_next_review": interval_days
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Review recording error: {e}")
        return json.dumps({
            "success": False,
            "error": f"Review recording failed: {str(e)}"
        })

@tool
def get_study_statistics() -> str:
    """
    Get comprehensive study statistics and learning analytics.
    
    Returns:
        JSON string with detailed study statistics, progress tracking, and recommendations
    """
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
                    COUNT(CASE WHEN next_review IS NOT NULL AND next_review <= %s THEN 1 END) as due_today
                FROM flashcards
            """, (datetime.now(),))
            overall = cursor.fetchone()
            
            # Category breakdown
            cursor.execute("""
                SELECT category, 
                       COUNT(*) as total_count, 
                       COUNT(CASE WHEN status = 'known' THEN 1 END) as mastered_count,
                       AVG(success_rate) as avg_success_rate
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
            
            # Recent performance (last 7 days)
            cursor.execute("""
                SELECT 
                    DATE(last_reviewed) as review_date,
                    COUNT(*) as cards_reviewed,
                    AVG(success_rate) as daily_success_rate
                FROM flashcards 
                WHERE last_reviewed >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY DATE(last_reviewed)
                ORDER BY review_date DESC
            """)
            recent_performance = cursor.fetchall()
            
            # Cards needing attention (low success rate)
            cursor.execute("""
                SELECT id, category, question, success_rate, review_count
                FROM flashcards 
                WHERE success_rate < 0.6 AND review_count >= 2
                ORDER BY success_rate ASC
                LIMIT 5
            """)
            struggling_cards = cursor.fetchall()
        
        store.close()
        
        # Generate recommendations
        recommendations = []
        if overall[6] > 0:  # due_today > 0
            recommendations.append(f"You have {overall[6]} cards due for review today!")
        
        if overall[3] > 10:  # new > 10
            recommendations.append(f"You have {overall[3]} new cards to learn. Consider starting with 5-10 cards per session.")
        
        avg_success = float(overall[5]) if overall[5] else 0
        if avg_success < 0.7:
            recommendations.append("Your overall success rate is below 70%. Focus on reviewing cards more frequently.")
        
        if study_streak == 0:
            recommendations.append("Start your study streak today! Even 5 minutes of daily practice helps.")
        elif study_streak < 7:
            recommendations.append(f"Great! You've studied {study_streak} days this month. Try to build a daily habit.")
        
        result = {
            "success": True,
            "overall_stats": {
                "total_cards": overall[0],
                "known": overall[1],
                "learning": overall[2],
                "new": overall[3],
                "average_reviews": round(float(overall[4]) if overall[4] else 0, 2),
                "average_success_rate": round(avg_success, 3),
                "due_today": overall[6]
            },
            "categories": [
                {
                    "category": cat[0],
                    "total": cat[1],
                    "mastered": cat[2],
                    "mastery_rate": round(cat[2] / cat[1] * 100, 1) if cat[1] > 0 else 0,
                    "avg_success_rate": round(float(cat[3]) if cat[3] else 0, 3)
                }
                for cat in categories
            ],
            "study_streak_days": study_streak,
            "recent_performance": [
                {
                    "date": str(perf[0]),
                    "cards_reviewed": perf[1],
                    "success_rate": round(float(perf[2]) if perf[2] else 0, 3)
                }
                for perf in recent_performance
            ],
            "struggling_cards": [
                {
                    "id": card[0],
                    "category": card[1],
                    "question": card[2][:100] + "..." if len(card[2]) > 100 else card[2],
                    "success_rate": round(float(card[3]) if card[3] else 0, 3),
                    "review_count": card[4]
                }
                for card in struggling_cards
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

# Updated tool registry with flashcard tools
TOOLS = [
    fetch_gnews, 
    search_articles, 
    query_knowledge_graph, 
    search_entity_relationships,
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
    
    # Add system message if not present
    messages = state["messages"]
    if not messages or not any(msg.type == "system" for msg in messages):
        system_msg = ChatPromptTemplate.from_template(
            settings.SYSTEM_PROMPT + "\n\n"
            "Available tools:\n"
            "- fetch_gnews: Get latest news articles\n"
            "- search_articles: Search stored articles semantically\n" 
            "- query_knowledge_graph: Query entity relationships\n"
            "- search_entity_relationships: Find relationships between specific entities\n"
            "- generate_flashcards_from_content: Create educational flashcards from various sources\n"
            "- get_flashcards_for_study: Retrieve flashcards for study sessions\n"
            "- record_flashcard_review: Record review results and update card status\n"
            "- get_study_statistics: Get comprehensive learning analytics\n\n"
            "Use tools when appropriate to provide accurate, up-to-date information and educational support."
        ).format()
        messages = [system_msg] + messages
    
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
    
    # Create tool lookup
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
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_decision, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# Create the compiled graph
agent_graph = create_agent_graph()