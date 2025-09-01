from fastapi import FastAPI
from retrieval.fetch_news import fetch_gnews
from retrieval.vector_store import VectorStore
from core.models import NewsArticle
from typing import List
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from agents.agent import SYSTEM_PROMPT, app_graph
from fastapi.responses import StreamingResponse, JSONResponse
import json


app = FastAPI()

@app.get("/search")
def search(query: str):
    results = fetch_gnews(query)['articles']

    store = VectorStore()
    articles: List[NewsArticle] = []

    for article_data in results:
        try:
            article = NewsArticle(
                id=article_data["id"],
                title=article_data["title"],
                content=article_data.get("content", ""),
                summary=article_data.get("description", ""),
                url=article_data["url"],
                source=article_data["source"]["name"],
                author=article_data.get("author"),
                published_date=datetime.fromisoformat(
                    article_data["publishedAt"].replace("Z", "+00:00")
                ),
            )
            store.insert_article(article)
            articles.append(article)
        except Exception as e:
            print(f"Skipping article due to error: {e}")

    store.close()
    return {"results": articles}


@app.post("/chat")
async def chat_with_agent(messages: list[str]):
    history = [SystemMessage(content=SYSTEM_PROMPT)]
    for m in messages:
        history.append(HumanMessage(content=m))

    def stream():
        for chunk in app_graph.stream({"messages": history}):
            # Each chunk is like {"agent": {"messages": [...]}} or {"tools": {...}}
            for node_name, state in chunk.items():
                if "messages" in state and state["messages"]:
                    ai_msg = state["messages"][-1]
                    if hasattr(ai_msg, "content") and ai_msg.content:
                        yield f"data: {json.dumps({'content': ai_msg.content})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

