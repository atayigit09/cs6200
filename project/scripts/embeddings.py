import argparse
import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch

from models import create_embedding_model
from rag.document_store import (
    Document, RAGDocumentStore, FaissVectorStore, ChromaVectorStore,
    load_documents_from_directory, split_document
)

@dataclass
class DocumentProcessor:
    """Class for processing documents from a topic directory."""
    topic: str
    docs_dir: str = "data/docs"
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    def get_topic_path(self) -> str:
        """Get the path to the topic directory."""
        return os.path.join(self.docs_dir, self.topic)
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the topic directory."""
        topic_path = self.get_topic_path()
        if not os.path.exists(topic_path):
            raise FileNotFoundError(f"Topic directory not found: {topic_path}")
        
        print(f"Loading documents from {topic_path}")
        return load_documents_from_directory(topic_path)
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting them into chunks.
        Returns a new list of chunked documents.
        """
        chunked_docs = []
        for doc in documents:
            # Split document into chunks
            chunks = split_document(
                doc, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            chunked_docs.extend(chunks)
        
        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

def load_model_config():
    """Loads the configuration file."""
    config_path = Path("configs/base_model.yaml").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found!")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Document Embedding Generation")
    
    parser.add_argument("--embedding_type", type=str, default="sentence_transformer",
                      help="Type of embedding model to use")
    
    parser.add_argument("--topic", type=str, required=True,
                      choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                      help="Topic directory to process documents from")
    
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                      help="Directory to save the vector database")
    
    args = parser.parse_args()
    
    # Load config dynamically
    args.model_config = load_model_config()
    
    return args

def initialize_vector_store(config: Dict[str, Any], topic: str, output_dir: str) -> RAGDocumentStore:
    """Initialize the vector store based on configuration."""
    rag_config = config.get('rag', {})
    vector_store_config = rag_config.get('vector_db', {})
    embedding_config = rag_config.get('embedding', {})
    vector_store_type = vector_store_config.get('type', 'faiss')
    embedding_dim = embedding_config.get('embedding_dimension', 768)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    topic_output_dir = os.path.join(output_dir, topic)
    os.makedirs(topic_output_dir, exist_ok=True)
    
    if vector_store_type.lower() == 'chroma':
        vector_store = ChromaVectorStore(
            collection_name=topic,
            persist_directory=topic_output_dir
        )
        print(f"Initialized Chroma vector store for topic '{topic}'")
    else:  # Default to FAISS
        vector_store = FaissVectorStore(dimension=embedding_dim)
        print(f"Initialized FAISS vector store with dimension {embedding_dim}")
    
    # Create the RAG document store with the vector store
    return RAGDocumentStore(vector_store=vector_store)

if __name__ == "__main__":
    opt = parse_args()
    
    # Initialize document processor
    processor = DocumentProcessor(
        topic=opt.topic,
        chunk_size=opt.model_config.get("rag").get("chunk_size"),
        chunk_overlap=opt.model_config.get("rag").get("chunk_overlap")
    )
    
    # Load and process documents
    original_docs = processor.load_documents()
    if not original_docs:
        print(f"No documents found for topic '{opt.topic}'")
        exit(1)
    
    chunked_docs = processor.process_documents(original_docs)


    # Load embedding model
    print("Loading embedding model...")
    embedding_model = create_embedding_model(opt)
    
    # Initialize vector store
    doc_store = initialize_vector_store(
        config=opt.model_config,
        topic=opt.topic,
        output_dir=opt.output_dir
    )
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(chunked_docs)} document chunks...")
    
    # Process in batches to avoid memory issues
    batch_size = 128
    for i in range(0, len(chunked_docs), batch_size):
        batch = chunked_docs[i:i+batch_size]
        
        # Extract text content for embedding
        texts = [doc.content for doc in batch]
        
        # Generate embeddings
        embeddings = embedding_model.embed(texts)
        
        # Add documents and embeddings to the document store
        doc_store.add_documents(batch, embeddings)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(chunked_docs)-1)//batch_size + 1}")
    
    # Save the document store with embeddings
    save_path = os.path.join(opt.output_dir, opt.topic)
    doc_store.save(save_path)
    print(f"Saved document store with embeddings to {save_path}")
    
    # If using FAISS, we need to explicitly save the vector store
    if isinstance(doc_store.vector_store, FaissVectorStore):
        doc_store.vector_store.save(os.path.join(save_path, "vector_store"))
    
    print("Document embedding generation complete!")