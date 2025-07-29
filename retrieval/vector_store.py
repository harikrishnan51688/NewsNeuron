import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import List, Dict, Any, Optional
from core.models import NewsArticle, QueryRequest
from retrieval.embeddings import EmbeddingGenerator
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.embedding_generator = EmbeddingGenerator()
        self._setup_database()
    
    def _setup_database(self):
        """Initialize database with required tables and extensions"""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    # Enable pgvector extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Create articles table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS articles (
                            id VARCHAR PRIMARY KEY,
                            title TEXT NOT NULL,
                            content TEXT NOT NULL,
                            summary TEXT,
                            url TEXT UNIQUE NOT NULL,
                            source VARCHAR NOT NULL,
                            author VARCHAR,
                            published_date TIMESTAMP NOT NULL,
                            categories TEXT[],
                            entities TEXT[],
                            embedding vector(1536),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Create index for vector similarity search
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS articles_embedding_idx 
                        ON articles USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    
                    # Create text search index
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS articles_text_search_idx 
                        ON articles USING gin(to_tsvector('english', title || ' ' || content));
                    """)
                    
                    conn.commit()
                    logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def store_article(self, article: NewsArticle) -> bool:
        """Store a single article in the vector database"""
        try:
            # Generate embedding if not provided
            if not article.embedding:
                content_for_embedding = f"{article.title} {article.summary or article.content[:500]}"
                article.embedding = self.embedding_generator.generate_embedding(content_for_embedding)
            
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO articles 
                        (id, title, content, summary, url, source, author, published_date, 
                         categories, entities, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            summary = EXCLUDED.summary,
                            updated_at = CURRENT_TIMESTAMP;
                    """, (
                        article.id, article.title, article.content, article.summary,
                        article.url, article.source, article.author, article.published_date,
                        article.categories, article.entities, article.embedding
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error storing article {article.id}: {e}")
            return False
    
    def semantic_search(self, query: str, limit: int = 10, threshold: float = 0.7) -> List[NewsArticle]:
        """Perform semantic search using vector similarity"""
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, title, content, summary, url, source, author, 
                               published_date, categories, entities,
                               1 - (embedding <=> %s::vector) as similarity_score
                        FROM articles 
                        WHERE 1 - (embedding <=> %s::vector) > %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                    """, (query_embedding, query_embedding, threshold, query_embedding, limit))
                    
                    results = []
                    for row in cur.fetchall():
                        article = NewsArticle(
                            id=row['id'],
                            title=row['title'],
                            content=row['content'],
                            summary=row['summary'],
                            url=row['url'],
                            source=row['source'],
                            author=row['author'],
                            published_date=row['published_date'],
                            categories=row['categories'] or [],
                            entities=row['entities'] or [],
                            relevance_score=row['similarity_score']
                        )
                        results.append(article)
                    
                    return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, limit: int = 10) -> List[NewsArticle]:
        """Perform traditional keyword search"""
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT id, title, content, summary, url, source, author, 
                               published_date, categories, entities,
                               ts_rank(to_tsvector('english', title || ' ' || content), 
                                      plainto_tsquery('english', %s)) as rank_score
                        FROM articles 
                        WHERE to_tsvector('english', title || ' ' || content) @@ 
                              plainto_tsquery('english', %s)
                        ORDER BY rank_score DESC
                        LIMIT %s;
                    """, (query, query, limit))
                    
                    results = []
                    for row in cur.fetchall():
                        article = NewsArticle(
                            id=row['id'],
                            title=row['title'],
                            content=row['content'],
                            summary=row['summary'],
                            url=row['url'],
                            source=row['source'],
                            author=row['author'],
                            published_date=row['published_date'],
                            categories=row['categories'] or [],
                            entities=row['entities'] or [],
                            relevance_score=row['rank_score']
                        )
                        results.append(article)
                    
                    return results
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []