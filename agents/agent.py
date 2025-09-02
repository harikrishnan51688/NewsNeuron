import os
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from core.config import settings
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from retrieval.fetch_news import fetch_gnews as fetch_gnews_api
import json
from retrieval.vector_store import VectorStore
from retrieval.knowledge_graph import KnowledgeGraph


vs = VectorStore()
kg = KnowledgeGraph()


MODEL = settings.AGENT_MODEL
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = settings.OPENROUTER_BASE_URL

SYSTEM_PROMPT = settings.SYSTEM_PROMPT

class ChatState(TypedDict):
    messages: List[BaseMessage]


# -------- Tools --------
@tool
def fetch_gnews(query: str) -> str:
    """
    Fetch news articles based on a search query.
    
    Args:
        query: The search term for news articles
        
    Returns:
        JSON string containing news articles with title, content, summary, url, source, etc.
    """
    try:
        results = fetch_gnews_api(query)
        return json.dumps(results, indent=2)
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return json.dumps({"error": str(e)}, indent=2)

@tool
def search_articles(query: str, top_k: int = 5) -> str:
    """
    Search stored articles by semantic similarity.
    Args:
        query: User search text
        top_k: Number of top matches to return
    Returns:
        JSON string with article details (from Postgres).
    """
    try:
        embedding = vs.embedding_generator.generate_embeddings(query)
        results = vs.pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True)

        articles = []
        with vs.pg_conn.cursor() as cursor:
            for match in results["matches"]:
                article_id = match["metadata"]["article_id"]
                cursor.execute("SELECT * FROM articles WHERE id = %s;", (article_id,))
                row = cursor.fetchone()
                if row:
                    # Map to dict
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
                        "similarity": match["score"]
                    })
        return json.dumps(articles, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

@tool
def query_knowledge_graph(question: str) -> dict:
    """
    Query the knowledge graph for relationships involving entities mentioned in the question.
    This tool can handle various types of questions:
    - Entity information: "What do we know about John Smith?"
    - Relationship queries: "What is the relationship between Company A and John Doe?"
    - Count queries: "How many connections does Paris have?"
    - General searches: "Tell me about the connections involving Tesla"
    
    Args:
        question: The natural language question to query the knowledge graph
        
    Returns:
        A dictionary containing:
        - success: Boolean indicating if the query was successful
        - query_type: Type of query performed (entity_information, relationship_search, etc.)
        - original_question: The input question
        - Various result fields depending on query type
        - error: Error message if something went wrong
    """
    return kg.analyze_question_and_query(question)

@tool
def search_relationships_between_entities(entity1: str, entity2: str) -> dict:
    """
    Search for direct relationships between two specific entities.
    
    Args:
        entity1: First entity name
        entity2: Second entity name
        
    Returns:
        Dictionary containing relationships between the two entities
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
        
        return {
            "success": True,
            "entity1": entity1,
            "entity2": entity2,
            "relationships_found": len(results),
            "relationships": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error searching relationships: {str(e)}"
        }

tools = [fetch_gnews, search_articles, query_knowledge_graph, search_relationships_between_entities]

# Create prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\n"
    "You have access to tools to help answer questions. "
    "- Use the fetch_gnews tool when users ask about news, current events, or specific topics they want news about."
    "- Use search_articles when the user ask about past articles or ask for making quizzes."
    "- Always use query_knowledge_graph to analyze questions and extract relevant information from the knowledge graph."
    "- Use search_relationships_between_entities to find direct relationships between two specific entities."),

    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- LangGraph Node ---
def call_model(state: ChatState) -> ChatState:
    llm = ChatOpenAI(
        model=MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        streaming=True,
    ).bind_tools(tools)

    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}

def call_tools(state: ChatState) -> ChatState:
    """Execute tool calls if the model requested them"""
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("Tool requested:", last_message.tool_calls)
        tool_results = []

        # Build a lookup of tool name → function
        tool_map = {t.name: t for t in tools}

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})

            if tool_name in tool_map:
                try:
                    # ✅ unpack args so {"query": "AI"} works
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Error calling {tool_name}: {e}"
            else:
                result = f"Unknown tool: {tool_name}"

            # Always return a ToolMessage with the matching id
            tool_results.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )

        return {"messages": state["messages"] + tool_results}

    return {"messages": state["messages"]}

def should_continue(state: ChatState) -> str:
    """Decide whether to continue with tools or end"""
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation
    return END

# -------- Build Graph --------
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

# Add edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)
workflow.add_edge("tools", "agent")

app_graph = workflow.compile()
