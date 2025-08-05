import psycopg2
from core.models import NewsArticle
from core.config import settings
from .embeddings import EmbeddingGenerator
from pinecone import Pinecone, ServerlessSpec
from .knowledge_graph import KnowledgeGraph

class VectorStore:
    def __init__(self):
        self.connection = None
        self._connect()
        self._setup_table()
    
    def _connect(self):
        """Establish a connection to the PostgreSQL database."""
        if self.connection is None:
            try:
                self.connection = psycopg2.connect(settings.POSTGRES_URL)
                self.connection.autocommit = True
            except Exception as e:
                raise ConnectionError(f"Failed to connect to the database: {e}")
    
    def _setup_table(self):
        """Create the articles table if it doesn't exist."""
        with self.connection.cursor() as cursor:

            # Enable the pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create the articles table with necessary fields
            dimension = settings.EMBEDDING_DIMENSION
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    summary TEXT,
                    url TEXT,
                    source TEXT,
                    author TEXT,
                    published_date TIMESTAMP,
                    categories TEXT[],
                    entities TEXT[],
                    embedding vector({dimension}),  
                    relevance_score FLOAT8
                );
            """)

            # Create index for vector similarity search
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
        self.connection.commit()
    
    def insert_article(self, article: NewsArticle) -> str:
        """Insert a NewsArticle into the vector store."""
        try:
            embedding_generator = EmbeddingGenerator()
            text = f"{article.title} {article.summary or article.content[:500]}"
            embedding = embedding_generator.generate_embeddings(text)

            if not embedding or len(embedding) != 1536:
                raise ValueError(f"Embedding must be 1536-dimensional, got {len(embedding)}")
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO articles 
                    (title, content, summary, url, source, author, published_date, 
                     categories, entities, embedding, relevance_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    article.title,
                    article.content,
                    article.summary,
                    article.url,
                    article.source,
                    article.author,
                    article.published_date,
                    article.categories,
                    article.entities,
                    embedding,
                    article.relevance_score
                ))
                article_id = cursor.fetchone()[0]

            # Generate embedding
            embedding_generator = EmbeddingGenerator()
            text = f"{article.title} {article.summary or article.content[:500]}"
            embedding = embedding_generator.generate_embeddings(text)

            if not embedding or len(embedding) != settings.EMBEDDING_DIMENSION:
                raise ValueError(f"Embedding must be {settings.EMBEDDING_DIMENSION}-dimensional, got {len(embedding)}")

            # Insert into Pinecone
            metadata = {
                "article_id": str(article_id),
                "title": article.title,
                "source": article.source,
                "published_date": str(article.published_date)
            }

            self.pinecone_index.upsert([(str(article_id), embedding, metadata)])

            # Add article to knowledge graph
            kg = KnowledgeGraph()
            try:
                text = f"{article.title} {article.summary or article.content[:500]}"
                kg.add_article_to_graph(text)
            except Exception as e:
                raise Exception(f"Error adding article to knowledge graph: {e}")
            finally:
                kg.close()

            return str(article_id)

        except Exception as e:
            self.connection.rollback()
            raise Exception(f"Failed to insert article: {e}")


            
            



        

        