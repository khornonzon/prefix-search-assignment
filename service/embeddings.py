
import os
from typing import List, Optional
import numpy as np


from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for generating embeddings for products and queries."""
    
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name 
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        # Get actual embedding dimension
        test_embedding = self.model.encode("test")
        self.EMBEDDING_DIM = len(test_embedding)
        print(f"Embedding dimension: {self.EMBEDDING_DIM}")
    
    def encode(self, texts: List[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        return self._encode_sentence_transformers(texts)
    
    def _encode_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model not initialized")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings
    
    def encode_product(self, name: str, description: str = "", keywords: str = "") -> np.ndarray:
        combined_text = f"{name}"
        if description:
            combined_text += f" {description}"
        if keywords:
            combined_text += f" {keywords}"
        
        return self.encode(combined_text)[0]


