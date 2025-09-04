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

from services.flashcard_service import FlashcardsService
from core.models import FlashcardRequest, SourceType

service = FlashcardsService()

@tool
def generate_flashcards_from_content(topic: str, source_type: str = "recent_news",
                                     difficulty: str = "medium", count: int = 10) -> str:
    """
    Generate, save, and return flashcards from news, knowledge base, or entities.
    """
    try:
        # Use the existing generation logic to get flashcards
        request = FlashcardRequest(
            topic=topic,
            source_type=SourceType(source_type),
            difficulty=difficulty,
            count=count
        )
        # Generate flashcards from news/entities/KB
        flashcards = json.loads(generate_flashcards_from_news(topic, source_type, difficulty, count))["flashcards"]

        # Save them into DB
        service.save_flashcards(flashcards, request)

        return json.dumps({
            "success": True,
            "flashcards": flashcards,
            "count": len(flashcards)
        })

    except Exception as e:
        logger.error(f"Error generating flashcards: {e}")
        return json.dumps({"success": False, "error": str(e)})
@tool
def record_flashcard_review(card_id: str, correct: bool, difficulty_rating: int = None) -> str:
    """
    Record user review for a specific flashcard and update spaced repetition stats.
    """
    try:
        result = service.record_review(card_id, correct, difficulty_rating)
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Review update failed: {e}")
        return json.dumps({"success": False, "error": str(e)})
@tool
def get_flashcard_stats() -> str:
    """
    Get overall flashcard statistics for progress tracking.
    """
    try:
        stats = service.get_flashcard_stats()
        return json.dumps({"success": True, "stats": stats})
    except Exception as e:
        logger.error(f"Error fetching flashcard stats: {e}")
        return json.dumps({"success": False, "error": str(e)})

# Updated tool registry
TOOLS = [
    fetch_gnews, 
    search_articles, 
    query_knowledge_graph, 
    generate_flashcards_from_content,
    record_flashcard_review,
    get_flashcard_stats
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