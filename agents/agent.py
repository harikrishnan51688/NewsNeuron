import os
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from core.config import settings
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, AgentType
from retrieval.fetch_news import fetch_gnews as fetch_gnews_api
import json


MODEL = settings.AGENT_MODEL
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY
OPENROUTER_BASE_URL = settings.OPENROUTER_BASE_URL

SYSTEM_PROMPT = (settings.SYSTEM_PROMPT or "You are a helpful assistant. Answer clearly and concisely.")

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


tools = [fetch_gnews]

# Create prompt template for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT + "\n\nYou have access to tools to help answer questions. Use the fetch_gnews tool when users ask about news, current events, or specific topics they want news about."),
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
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("Tool requested:", last_message.tool_calls)
        tool_results = []
        
        for tool_call in last_message.tool_calls:
            # Execute the tool
            if tool_call["name"] == "fetch_gnews":
                try:
                    result = fetch_gnews.invoke(tool_call["args"])
                    tool_results.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call["id"]
                        )
                    )
                except Exception as e:
                    tool_results.append(
                        ToolMessage(
                            content=f"Error calling tool: {str(e)}",
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
