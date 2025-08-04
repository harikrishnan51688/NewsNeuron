import psycopg2
from core.models import NewsArticle
from core.config import settings


class VectorStore:
    def __init__(self):
        self.connection = None
        self._connect()
    
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
        self._connect()
        with self.connection.cursor() as cursor:

            # Enable the pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create the articles table with necessary fields
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
                    embedding vector(1536),  
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
            
            



        

        