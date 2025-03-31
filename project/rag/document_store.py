from pathlib import Path
import os
import json
import pickle
from typing import Dict, List, Optional, Union, Any
import numpy as np
 
class Document:
    """A class representing a document with content and metadata."""
   
    def __init__(self, content: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Document object.
       
        Args:
            content: The text content of the document
            doc_id: Unique identifier for the document
            metadata: Additional information about the document
        """
        self.content = content
        self.doc_id = doc_id
        self.metadata = metadata or {}
   
    def __repr__(self):
        return f"Document(id={self.doc_id}, metadata={self.metadata})"
 
 
class DocumentStore:
    """Base class for document storage and retrieval."""
   
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store."""
        raise NotImplementedError
   
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        raise NotImplementedError
   
    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for documents relevant to the query."""
        raise NotImplementedError
 
 
class InMemoryDocumentStore(DocumentStore):
    """Simple in-memory document store."""
   
    def __init__(self):
        self.documents = {}  # doc_id -> Document
   
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the store."""
        doc_ids = []
        for doc in documents:
            if doc.doc_id is None:
                doc.doc_id = str(len(self.documents))
            self.documents[doc.doc_id] = doc
            doc_ids.append(doc.doc_id)
        return doc_ids
   
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        return self.documents.get(doc_id)
   
    def get_all_documents(self) -> List[Document]:
        """Get all documents in the store."""
        return list(self.documents.values())
 
 
class VectorStore:
    """Base class for vector databases."""
   
    def add_embeddings(self, doc_ids: List[str], embeddings: List[List[float]]) -> None:
        """Add embeddings for documents."""
        raise NotImplementedError
   
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """Search for similar embeddings."""
        raise NotImplementedError
   
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        raise NotImplementedError
   
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """Load a vector store from disk."""
        raise NotImplementedError
 
 
class FaissVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search."""
   
    def __init__(self, dimension: int = 768):
        """
        Initialize a FAISS vector store.
       
        Args:
            dimension: Dimensionality of the embeddings
        """
        try:
            import faiss
            self.dimension = dimension
            self.index = faiss.IndexFlatL2(dimension)
            self.doc_ids = []
        except ImportError:
            raise ImportError("FAISS is not installed. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'.")
   
    def add_embeddings(self, doc_ids: List[str], embeddings: List[List[float]]) -> None:
        """
        Add embeddings for documents.
       
        Args:
            doc_ids: List of document IDs
            embeddings: List of embedding vectors
        """
        if len(doc_ids) != len(embeddings):
            raise ValueError("Number of document IDs must match number of embeddings.")
       
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.doc_ids.extend(doc_ids)
   
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Search for similar embeddings.
       
        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
           
        Returns:
            List of dictionaries with document IDs and similarity scores
        """
        if len(self.doc_ids) == 0:
            return []
       
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, min(top_k, len(self.doc_ids)))
       
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.doc_ids):  # Ensure index is valid
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "score": float(distances[0][i])
                })
       
        return results
   
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
       
        Args:
            path: Directory to save to
        """
        import faiss
        os.makedirs(os.path.dirname(path), exist_ok=True)
       
        # Save the index
        index_path = f"{path}.index"
        faiss.write_index(self.index, index_path)
       
        # Save the document IDs and dimension
        metadata = {
            "doc_ids": self.doc_ids,
            "dimension": self.dimension
        }
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(metadata, f)
   
    @classmethod
    def load(cls, path: str) -> 'FaissVectorStore':
        """
        Load a vector store from disk.
       
        Args:
            path: Path to the saved vector store
           
        Returns:
            Loaded FaissVectorStore
        """
        import faiss
       
        # Load metadata
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)
       
        # Create a new instance
        store = cls(dimension=metadata["dimension"])
       
       
        # Load the index
        index_path = f"{path}.index"
        store.index = faiss.read_index(index_path)
       
        # Set the document IDs
        store.doc_ids = metadata["doc_ids"]
       
        return store
 
 
class ChromaVectorStore(VectorStore):
    """Chroma-based vector store for persistent storage."""
   
    def __init__(self, collection_name: str = "default", persist_directory: Optional[str] = None):
        """
        Initialize a Chroma vector store.
       
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        try:
            import chromadb
            self.client = chromadb.Client(
                chromadb.config.Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False
                )
            )
            self.collection = self.client.get_or_create_collection(collection_name)
        except ImportError:
            raise ImportError("Chroma is not installed. Please install it with 'pip install chromadb'.")
   
    def add_embeddings(self, doc_ids: List[str], embeddings: List[List[float]]) -> None:
        """
        Add embeddings for documents.
       
        Args:
            doc_ids: List of document IDs
            embeddings: List of embedding vectors
        """
        if len(doc_ids) != len(embeddings):
            raise ValueError("Number of document IDs must match number of embeddings.")
       
        # ChromaDB needs documents, but we're just using embeddings
        # Create placeholder documents (empty strings)
        documents = [""] * len(doc_ids)
       
        # Add in batches to handle large numbers of documents
        batch_size = 100
        for i in range(0, len(doc_ids), batch_size):
            end_idx = min(i + batch_size, len(doc_ids))
            self.collection.add(
                ids=doc_ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx]
            )
   
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
        """
        Search for similar embeddings.
       
        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
           
        Returns:
            List of dictionaries with document IDs and similarity scores
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
       
        if not results["ids"] or not results["ids"][0]:
            return []
       
        # Format the results
        formatted_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            formatted_results.append({
                "doc_id": doc_id,
                "score": float(results["distances"][0][i]) if "distances" in results else 0.0
            })
       
        return formatted_results
   
    def save(self, path: str = None) -> None:
        """
        Persist the vector store.
       
        Args:
            path: Not used for Chroma as it persists automatically
        """
        # Chroma persists automatically if persist_directory is set
        if hasattr(self.client, "persist"):
            self.client.persist()
   
    @classmethod
    def load(cls, path: str, collection_name: str = "default") -> 'ChromaVectorStore':
        """
        Load a vector store.
       
        Args:
            path: Path to the persisted database
            collection_name: Name of the collection
           
        Returns:
            Loaded ChromaVectorStore
        """
        return cls(collection_name=collection_name, persist_directory=path)
 
 
class RAGDocumentStore:
    """
    Document store with vector search capabilities for RAG.
    Combines document storage and vector search.
    """
   
    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize a RAG document store.
       
        Args:
            document_store: Storage for document content
            vector_store: Storage for document embeddings
        """
        self.document_store = document_store or InMemoryDocumentStore()
        self.vector_store = vector_store
   
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
       
        # Add to document store
        doc_ids = self.document_store.add_documents(documents)
       
        # Add to vector store if available
        if self.vector_store is not None:
            self.vector_store.add_embeddings(doc_ids, embeddings)
   
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """Search for similar documents using the query embedding."""
        if self.vector_store is None:
            return []
       
        # Get similar documents from vector store
        results = self.vector_store.search(query_embedding, top_k)
       
        # Fetch full documents
        documents = []
        for result in results:
            doc = self.document_store.get_document_by_id(result["doc_id"])
            if doc:
                doc.metadata["score"] = result["score"]
                documents.append(doc)
       
        return documents
   
    def save(self, path: str) -> None:
        """Save the document store to disk."""
        os.makedirs(path, exist_ok=True)
       
        # Save documents
        documents_data = {}
        for doc in self.document_store.get_all_documents():
            documents_data[doc.doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata
            }
       
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(documents_data, f)
       
        # Save vector store if available
        if self.vector_store is not None:
            self.vector_store.save(os.path.join(path, "vector_store"))
   
    @classmethod
    def load(cls, path: str, vector_store_class: type = FaissVectorStore) -> 'RAGDocumentStore':
        """Load a document store from disk."""
        document_store = InMemoryDocumentStore()
       
        # Load documents
        with open(os.path.join(path, "documents.json"), "r") as f:
            documents_data = json.load(f)
       
        documents = []
        for doc_id, data in documents_data.items():
            document = Document(
                content=data["content"],
                doc_id=doc_id,
                metadata=data["metadata"]
            )
            documents.append(document)
       
        document_store.add_documents(documents)
       
        # Load vector store
        vector_store_path = os.path.join(path, "vector_store")
        if os.path.exists(f"{vector_store_path}.meta"):
            vector_store = vector_store_class.load(vector_store_path)
        else:
            vector_store = None
       
        return cls(document_store=document_store, vector_store=vector_store)
 
 
def load_documents_from_directory(directory: str, extensions: List[str] = [".txt", ".md"]) -> List[Document]:
    """
    Load documents from files in a directory.
   
    Args:
        directory: Path to the directory
        extensions: List of file extensions to include
       
    Returns:
        List of Document objects
    """
    directory_path = Path(directory)
    documents = []
   
    for ext in extensions:
        for file_path in directory_path.glob(f"**/*{ext}"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
               
                # Create metadata with file information
                relative_path = file_path.relative_to(directory_path)
                metadata = {
                    "source": str(relative_path),
                    "filename": file_path.name,
                    "created_at": os.path.getctime(file_path)
                }
               
                # Create document
                document = Document(
                    content=content,
                    metadata=metadata
                )
                documents.append(document)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
   
    return documents
 
 
def split_document(document: Document, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split a document into chunks.
   
    Args:
        document: Document to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between consecutive chunks
       
    Returns:
        List of Document chunks
    """
    content = document.content
    chunks = []
   
    # Skip chunking for very short documents
    if len(content) <= chunk_size:
        # Just create a single chunk with the original content
        chunk_doc = Document(
            content=content,
            metadata={
                **document.metadata,
                "chunk_id": 0,
                "original_doc_id": document.doc_id
            }
        )
        chunks.append(chunk_doc)
        return chunks
    
    # Use a more robust chunking approach
    # Use sentence or paragraph boundaries when possible
    start_idx = 0
    seen_content = set()  # Track content we've already added
    
    while start_idx < len(content):
        # Calculate end index for this chunk
        end_idx = min(start_idx + chunk_size, len(content))
        
        # Get chunk content
        chunk_content = content[start_idx:end_idx]
        
        # Skip if chunk is too small or duplicate
        if len(chunk_content) < 50 or chunk_content in seen_content:
            start_idx = start_idx + chunk_size - chunk_overlap
            continue
        
        # Add to seen content
        seen_content.add(chunk_content)
        
        # Create chunk document
        chunk_doc = Document(
            content=chunk_content,
            metadata={
                **document.metadata,
                "chunk_id": len(chunks),
                "chunk_start": start_idx,
                "chunk_end": end_idx,
                "original_doc_id": document.doc_id
            }
        )
        chunks.append(chunk_doc)
        
        # Move to next chunk position with overlap
        start_idx = start_idx + chunk_size - chunk_overlap
   
    return chunks
