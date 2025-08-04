from core.config import Settings
from typing import List, Dict, Any

class EmbeddingGenerator:
    def __init__(self, model: str = Settings.EMBEDDING_MODEL, dimension: int = Settings.EMBEDDING_DIMENSION):
        self.model = model
        self.dimension = dimension

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the specified model.
        """
        
