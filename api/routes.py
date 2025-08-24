from fastapi import FastAPI
from retrieval.fetch_news import fetch_gnews
from retrieval.vector_store import VectorStore
from core.models import NewsArticle
from typing import List
from datetime import datetime

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