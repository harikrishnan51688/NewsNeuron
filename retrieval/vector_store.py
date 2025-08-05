import psycopg2
from core.models import NewsArticle
from core.config import settings
from .embeddings import EmbeddingGenerator
from pinecone import Pinecone, ServerlessSpec
from .knowledge_graph import KnowledgeGraph

class VectorStore:
    def __init__(self):
        self.pg_conn = None
        self.pinecone_index = None
        self._connect_postgres()
        self._setup_postgres_table()
        self._init_pinecone()
    
    def _connect_postgres(self):
        """Connect to PostgreSQL database."""
        if self.pg_conn is None:
            try:
                self.pg_conn = psycopg2.connect(settings.POSTGRES_URL)
                self.pg_conn.autocommit = True
            except Exception as e:
                raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")
    
    def _setup_postgres_table(self):
        """Create the articles table (excluding embedding)."""
        with self.pg_conn.cursor() as cursor:
            cursor.execute("""
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
                    relevance_score FLOAT8
                );
            """)
        self.pg_conn.commit()
        
    def _init_pinecone(self):
        """Initialize Pinecone vector index."""
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT)
            )
        # Connect to the Pinecone index
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

    def insert_article(self, article: NewsArticle) -> str:
        try:
            # Insert into PostgreSQL
            with self.pg_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO articles 
                    (title, content, summary, url, source, author, published_date, 
                     categories, entities, relevance_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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


            
            



        

        