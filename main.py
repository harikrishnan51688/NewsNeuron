from retrieval.vector_store import VectorStore
from core.models import NewsArticle
from uuid import uuid4
from retrieval.knowledge_graph import KnowledgeGraph

article_data = {
    "id": str(uuid4()),
    "title": "Netflix reports record subscriber growth amid streaming wars intensify",
    "content": "Netflix added 8.2 million subscribers in Q3 2023, surpassing analyst expectations of 6 million. The growth was driven by popular original series and aggressive expansion into gaming. CEO Reed Hastings announced plans to invest an additional $2 billion in content production for 2024, despite increased competition from Disney+ and Amazon Prime.",
    "summary": "Netflix beats expectations with 8.2M new subscribers, plans $2B content investment.",
    "url": "http://example.com/netflix-q3-earnings-2023",
    "source": "Business Wire",
    "author": "Jennifer Park",
    "published_date": "2023-10-19T14:20:00Z",
    "categories": ["business", "entertainment", "streaming", "earnings"],
    "entities": ["Netflix", "Reed Hastings", "Disney+", "Amazon Prime", "streaming"],
    "relevance_score": 0.79
}

article = NewsArticle(**article_data)

store = VectorStore()
store.insert_article(article)
