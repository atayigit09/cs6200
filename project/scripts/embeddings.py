import argparse
import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

from models import create_embedding_model
from rag.document_store import (
    Document, RAGDocumentStore, FaissVectorStore, ChromaVectorStore,
    load_documents_from_directory, split_document
)

@dataclass
class DocumentProcessor:
    """Class for processing documents from a field directory."""
    field: str
    chunk_size: int
    chunk_overlap: int
    docs_dir: str = "data/docs"
    def get_field_path(self) -> str:
        """Get the path to the field directory."""
        return os.path.join(self.docs_dir, self.field)
    
    def load_documents(self) -> List[Document]:
        """Load all documents from the field directory."""
        field_path = self.get_field_path()
        if not os.path.exists(field_path):
            raise FileNotFoundError(f"field directory not found: {field_path}")
        
        print(f"Loading documents from {field_path}")
        return load_documents_from_directory(field_path)
    
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


def initialize_vector_store(config: Dict[str, Any], field: str, output_dir: str) -> RAGDocumentStore:
    """Initialize the vector store based on configuration."""
    rag_config = config.get('rag', {})
    vector_store_config = rag_config.get('vector_db', {})
    embedding_config = rag_config.get('embedding', {})
    vector_store_type = vector_store_config.get('type')
    embedding_dim = embedding_config.get('embedding_dimension')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    topic_output_dir = os.path.join(output_dir, field)
    os.makedirs(topic_output_dir, exist_ok=True)
    
    if vector_store_type.lower() == 'chroma':
        vector_store = ChromaVectorStore(
            collection_name=field,
            persist_directory=topic_output_dir
        )
        print(f"Initialized Chroma vector store for topic '{field}'")
    else:  # Default to FAISS
        vector_store = FaissVectorStore(dimension=embedding_dim)
        print(f"Initialized FAISS vector store with dimension {embedding_dim}")
    
    # Create the RAG document store with the vector store
    return RAGDocumentStore(vector_store=vector_store)


def load_model_config():
    """Loads the configuration file based on the model_name."""
    config_path = Path(f"configs/base_model.yaml").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found!")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Document Embedding Generation")
    
    parser.add_argument("--data_path", type=str, default="data/docs",
                      help="directory containing the documents")
    
    
    args = parser.parse_args()
    
    # Load config dynamically
    args.model_config = load_model_config()
    
    return args


if __name__ == "__main__":
    opt = parse_args()
    
    # Initialize document processor
    processor = DocumentProcessor(
        field=opt.model_config.get("rag").get("field"),
        chunk_size=opt.model_config.get("rag").get("chunk_size"),
        chunk_overlap=opt.model_config.get("rag").get("chunk_overlap")
    )
    
    # Load and process documents
    original_docs = processor.load_documents()
    if not original_docs:
        print(f"No documents found for field '{opt.model_config.get("rag").get("field")}'")
        exit(1)
    
    chunked_docs = processor.process_documents(original_docs)

    print("Original Docs length")
    print(len(original_docs))

    print("Chunked Docs length")
    print(len(chunked_docs))

    ##not the fastest way to remove duplicates but i'm tired and it works :)
    #for every doc in chunked_docs, create set of unique doc.content
    content_set = set()
    unique_chunked_docs = []
    
    # Safer way to deduplicate - build a new list instead of removing during iteration
    for doc in chunked_docs:
        if doc.content not in content_set:
            content_set.add(doc.content)
            unique_chunked_docs.append(doc)
    
    # Replace the original list with the deduplicated one
    chunked_docs = unique_chunked_docs

    print("Chunked Docs length after removing duplicates")
    print(len(chunked_docs))
    

    # Load embedding model
    print("Loading embedding model...")
    embedding_model = create_embedding_model(opt.model_config.get("rag").get("embedding"))
    
    # Initialize vector store
    doc_store = initialize_vector_store(
        config=opt.model_config,
        field=opt.model_config.get("rag").get("field"),
        output_dir=opt.model_config.get("rag").get("vector_db").get("db_path")   
    )
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(chunked_docs)} document chunks...")
    
    # Process in batches to avoid memory issues
    batch_size = opt.model_config.get("rag").get("batch_size")
    
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
    save_path = os.path.join(opt.model_config.get("rag").get("vector_db").get("db_path"), opt.model_config.get("rag").get("field"))
    doc_store.save(save_path)
    print(f"Saved document store with embeddings to {save_path}")
    
    
    print("Document embedding generation complete!")