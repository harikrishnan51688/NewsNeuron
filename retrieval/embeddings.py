from core.config import settings
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    def __init__(self, model: str = settings.EMBEDDING_MODEL, dimension: int = settings.EMBEDDING_DIMENSION):
        self.model_name = model
        self.dimension = dimension
        self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for a list of texts using the specified model.
        """
        try:
            embeddings = self.model.encode(text, convert_to_tensor=False).tolist()
            return embeddings
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {e}")
            
        
