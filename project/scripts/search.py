import argparse
import yaml
import os
from pathlib import Path
from typing import List, Dict, Any
import torch

from models import create_embedding_model
from rag.document_store import (
    Document, RAGDocumentStore, FaissVectorStore, ChromaVectorStore
)

def load_model_config():
    """Loads the configuration file."""
    config_path = Path("configs/base_model.yaml").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found!")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Search Vector Database")
    
    parser.add_argument("--embedding_type", type=str, default="sentence_transformer",
                      help="Type of embedding model to use (should match what was used for embedding generation)")
    
    parser.add_argument("--topic", type=str, required=True,
                      choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                      help="Topic to search within")
    
    
    parser.add_argument("--num_results", type=int, default=5,
                      help="Number of results to return")
    
    parser.add_argument("--embeddings_dir", type=str, default="data/embeddings",
                      help="Directory where vector database is stored")
    
    args = parser.parse_args()
    
    # Load config dynamically
    args.model_config = load_model_config()
    
    return args

def load_document_store(config: Dict[str, Any], topic: str, embeddings_dir: str) -> RAGDocumentStore:
    """Load the document store with vector database for a given topic."""
    rag_config = config.get('rag', {})
    vector_store_type = rag_config.get('vector_store_type', 'faiss')
    embedding_dim = rag_config.get('embedding_dimension', 768)
    
    topic_dir = os.path.join(embeddings_dir, topic)
    
    if not os.path.exists(topic_dir):
        raise FileNotFoundError(f"No embeddings found for topic '{topic}' at {topic_dir}")
    
    if vector_store_type.lower() == 'chroma':
        # For Chroma, we load the collection
        vector_store = ChromaVectorStore(
            collection_name=topic,
            persist_directory=topic_dir
        )
        doc_store = RAGDocumentStore(vector_store=vector_store)
        print(f"Loaded Chroma vector store for topic '{topic}'")
    else:
        # For FAISS, we load the index
        vector_store = FaissVectorStore(dimension=embedding_dim)
        vector_store = vector_store.load(os.path.join(topic_dir, "vector_store"))
        doc_store = RAGDocumentStore(vector_store=vector_store)
        doc_store = doc_store.load(topic_dir)
        print(f"Loaded FAISS vector store for topic '{topic}'")
    
    return doc_store

def format_search_result(doc: Document, score: float) -> str:
    """Format a search result for display."""
    header = f"Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}"
    divider = "-" * len(header)
    
    return f"{header}\n{divider}\n{doc.content}\n"

def search(query: str, doc_store: RAGDocumentStore, embedding_model, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the document store for documents similar to the query."""
    # Embed the query
    query_embedding = embedding_model.embed([query])[0]
    # Search for similar documents
    results = doc_store.search(query_embedding, top_k=num_results)
    
    return results

if __name__ == "__main__":
    opt = parse_args()
    
    # Load document store with vector database
    doc_store = load_document_store(
        config=opt.model_config,
        topic=opt.topic,
        embeddings_dir=opt.embeddings_dir
    )
    
    print(len(doc_store.vector_store.doc_ids))
    # Load embedding model (same as used for embedding creation)
    print("Loading embedding model...")
    embedding_model = create_embedding_model(opt)
    
    # Perform search

    query = "Can you clarify whether the claim \"10% of sudden infant death syndrome (SIDS) deaths happen in newborns aged less than 6 months.\" is accurate or not? Build factual arguments about the claim."
 
    results = search(
        query=query,
        doc_store=doc_store,
        embedding_model=embedding_model,
        num_results=opt.num_results
    )
    
    # Display results
    if not results:
        print("No matching documents found.")
    else:
        print(f"\nFound {len(results)} results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            doc = result.content
            score = result.metadata["score"]
            
            print(f"\nResult {i}:")
            print(format_search_result(result, score))
            
            if i < len(results):
                print("=" * 80)