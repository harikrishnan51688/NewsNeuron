"""
AI Agent with LangGraph for news and knowledge graph queries.
"""
import json
import logging
from typing import List, TypedDict, Literal

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
# TOOLS
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

# Tool registry
TOOLS = [fetch_gnews, search_articles, query_knowledge_graph, search_entity_relationships]

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
            "- search_entity_relationships: Find relationships between specific entities\n\n"
            "Use tools when appropriate to provide accurate, up-to-date information."
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