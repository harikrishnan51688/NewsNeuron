from typing import List, Dict, Any
import time
from core.models import NewsArticle, QueryRequest, RetrievalResult, QueryType
from retrieval.vector_store import VectorStore
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class NewsRetriever:
    def __init__(self):
        self.vector_store = VectorStore(settings.POSTGRES_URL)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user query to determine search strategy"""
        analysis = {
            "query_type": QueryType.FACTUAL,
            "entities": [],
            "keywords": [],
            "time_range": None,
            "use_semantic": True,
            "use_keyword": False
        }
        
        # Simple heuristics for query analysis
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ["how", "why", "explain", "analyze"]):
            analysis["query_type"] = QueryType.ANALYTICAL
        elif any(word in query_lower for word in ["when", "timeline", "chronology", "sequence"]):
            analysis["query_type"] = QueryType.TIMELINE
        elif any(word in query_lower for word in ["relationship", "connection", "between", "link"]):
            analysis["query_type"] = QueryType.RELATIONSHIP
        
        # Determine search strategy
        if len(query.split()) <= 3:
            analysis["use_keyword"] = True
        
        return analysis
    
    def retrieve_news(self, request: QueryRequest) -> RetrievalResult:
        """Main method to retrieve news based on user query"""
        start_time = time.time()
        
        try:
            # Analyze the query
            query_analysis = self.analyze_query(request.query)
            
            # Override query type if provided
            if request.query_type:
                query_analysis["query_type"] = request.query_type
            
            # Perform retrieval based on analysis
            articles = []
            
            if query_analysis["use_semantic"]:
                semantic_results = self.vector_store.semantic_search(
                    request.query, 
                    limit=request.limit,
                    threshold=settings.SIMILARITY_THRESHOLD
                )
                articles.extend(semantic_results)
            
            if query_analysis["use_keyword"] and len(articles) < request.limit:
                keyword_results = self.vector_store.keyword_search(
                    request.query,
                    limit=request.limit - len(articles)
                )
                
                # Merge results, avoiding duplicates
                existing_urls = {article.url for article in articles}
                for article in keyword_results:
                    if article.url not in existing_urls:
                        articles.append(article)
            
            # Apply filters if provided
            if request.filters:
                articles = self._apply_filters(articles, request.filters)
            
            # Sort by relevance score
            articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            
            # Limit results
            articles = articles[:request.limit]
            
            processing_time = time.time() - start_time
            
            return RetrievalResult(
                articles=articles,
                total_count=len(articles),
                query_analysis=query_analysis,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error retrieving news: {e}")
            return RetrievalResult(
                articles=[],
                total_count=0,
                query_analysis={},
                processing_time=time.time() - start_time
            )
    
    def _apply_filters(self, articles: List[NewsArticle], filters: Dict[str, Any]) -> List[NewsArticle]:
        """Apply filters to the retrieved articles"""
        filtered_articles = articles
        
        # Source filter
        if "source" in filters:
            sources = filters["source"] if isinstance(filters["source"], list) else [filters["source"]]
            filtered_articles = [a for a in filtered_articles if a.source in sources]
        
        # Date range filter
        if "date_from" in filters or "date_to" in filters:
            if "date_from" in filters:
                filtered_articles = [a for a in filtered_articles if a.published_date >= filters["date_from"]]
            if "date_to" in filters:
                filtered_articles = [a for a in filtered_articles if a.published_date <= filters["date_to"]]
        
        # Category filter
        if "categories" in filters:
            categories = filters["categories"] if isinstance(filters["categories"], list) else [filters["categories"]]
            filtered_articles = [a for a in filtered_articles if any(cat in a.categories for cat in categories)]
        
        return filtered_articles

# ============================================================================

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from datetime import datetime, timedelta
    
    # Example: Create and store some sample articles
    sample_articles = [
        NewsArticle(
            id="1",
            title="AI Revolution in Healthcare",
            content="Artificial intelligence is transforming healthcare with new diagnostic tools...",
            summary="AI is revolutionizing medical diagnosis and treatment",
            url="https://example.com/ai-healthcare",
            source="TechNews",
            published_date=datetime.now(),
            categories=["technology", "healthcare"],
            entities=["AI", "healthcare", "diagnosis"]
        ),
        NewsArticle(
            id="2",
            title="Climate Change Impact on Agriculture",
            content="Rising temperatures and changing weather patterns are affecting crop yields...",
            summary="Climate change is significantly impacting global agriculture",
            url="https://example.com/climate-agriculture",
            source="ScienceDaily",
            published_date=datetime.now() - timedelta(days=1),
            categories=["environment", "agriculture"],
            entities=["climate change", "agriculture", "crops"]
        )
    ]
    
    # Initialize retriever
    retriever = NewsRetriever()
    
    # Store sample articles
    for article in sample_articles:
        retriever.vector_store.store_article(article)
    
    # Test retrieval
    request = QueryRequest(
        query="artificial intelligence in medicine",
        limit=5
    )
    
    result = retriever.retrieve_news(request)
    
    print(f"Found {result.total_count} articles")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Query analysis: {result.query_analysis}")
    
    for article in result.articles:
        print(f"- {article.title} (Score: {article.relevance_score:.3f})")