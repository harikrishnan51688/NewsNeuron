from retrieval.vector_store import VectorStore
from core.models import NewsArticle
from uuid import uuid4


article_data = {
    "id": str(uuid4()),
    "title": "Test Article",
    "content": "This is the content of the test article.",
    "summary": "This is a summary of the test article.",
    "url": "http://example.com/test-article",
    "source": "Test Source",
    "author": "Test Author",
    "published_date": "2023-10-01T12:00:00Z",
    "categories": ["test", "article"],
    "entities": ["test", "example"],
    "relevance_score": 0.95
}

article = NewsArticle(**article_data)

store = VectorStore()
store.insert_article(article)
