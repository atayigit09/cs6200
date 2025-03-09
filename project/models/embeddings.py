from typing import Dict, List, Any, Union, Optional
import numpy as np
import os
import requests
import time
from models import EmbeddingModel
from sentence_transformers import SentenceTransformer
import openai



class SentenceTransformerEmbeddings(EmbeddingModel):
    """Embedding model using Sentence Transformers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        device = config.get("device", "cpu") 
        self.model = SentenceTransformer(config.get("model_name"), device=device)
    
    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        
        embeddings = self.model.encode(text)
        return embeddings.tolist()


class OpenAIEmbeddings(EmbeddingModel):
    """Embedding model using OpenAI's API."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
       
            
        self.openai = openai
        self.openai.api_key = config.get("api_key")
        self.model = config.get("model_name")

    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        
        # Create embeddings with rate limiting retry logic
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.openai.embeddings.create(
                    model=self.model,
                    input=text
                )
                
                # Extract embeddings from response
                embeddings = [item.embedding for item in response.data]
                return embeddings
                
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff
                    sleep_time = retry_delay * (2 ** attempt)
                    print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e


# class HuggingFaceEmbeddings(EmbeddingModel):
#     """Embedding model using Hugging Face's API."""
    
#     def __init__(self, api_key: Optional[str] = None, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         """
#         Initialize a Hugging Face embedding model.
        
#         Args:
#             api_key: Hugging Face API key (falls back to HF_API_KEY environment variable)
#             model_id: ID of the model to use
#         """
#         self.api_key = api_key or os.getenv("HF_API_KEY")
#         if not self.api_key:
#             raise ValueError("Hugging Face API key must be provided or set as HF_API_KEY environment variable")
#         self.model_id = model_id
#         self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
#         self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
#     def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
#         """
#         Generate embeddings for text.
        
#         Args:
#             text: Text or list of texts to embed
            
#         Returns:
#             List of embedding vectors
#         """
#         if isinstance(text, str):
#             text = [text]
        
#         # Create embeddings with rate limiting retry logic
#         max_retries = 5
#         retry_delay = 1
        
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     self.api_url,
#                     headers=self.headers,
#                     json={"inputs": text, "options": {"wait_for_model": True}}
#                 )
                
#                 if response.status_code != 200:
#                     response.raise_for_status()
                
#                 # Extract embeddings from response
#                 embeddings = response.json()
#                 if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
#                     return embeddings
#                 else:
#                     raise ValueError(f"Unexpected response format: {embeddings}")
                
#             except requests.exceptions.RequestException as e:
#                 if attempt < max_retries - 1:
#                     # Exponential backoff
#                     sleep_time = retry_delay * (2 ** attempt)
#                     print(f"API error: {e}. Retrying in {sleep_time} seconds...")
#                     time.sleep(sleep_time)
#                 else:
#                     raise e

