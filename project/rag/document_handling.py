from typing import Dict, List, Optional, Union, Any
import os
import pinecone
from pathlib import Path
import tiktoken
import re
from tqdm import tqdm
import numpy as np

class Document:
    """A class representing a document with content and metadata."""
    def __init__(self, content: str, doc_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.doc_id = doc_id
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, metadata={self.metadata})"

class DocumentProcessor:
    """Handles document chunking and preparation for embedding."""
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 120, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_text(self, document: Document) -> List[Document]:
        """
        Smart chunking strategy optimized for Wikipedia articles:
        - Preserves section headers with their content
        - Maintains paragraph coherence
        - Uses sliding window for context retention
        """
        text = document.content
        
        # Extract title if it exists (Wikipedia format: "# Title\n\nContent")
        title = None
        if text.startswith("# "):
            title_end = text.find("\n\n")
            if title_end != -1:
                title = text[2:title_end].strip()
                text = text[title_end + 2:]

        # Identify section headers (Wikipedia format)
        section_pattern = re.compile(r'^=+ .+ =+$', re.MULTILINE)
        sections = {match.start(): match.group(0).strip('= ') 
                   for match in section_pattern.finditer(text)}
        
        tokens = self.encoding.encode(text)
        chunks = []
        current_section = None
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Track current section for context
            for pos in sections:
                if i <= pos < i + self.chunk_size:
                    current_section = sections[pos]
            
            # Add title and section context
            prefix = []
            if title:
                prefix.append(f"Title: {title}")
            if current_section:
                prefix.append(f"Section: {current_section}")
            if prefix:
                chunk_text = f"{' | '.join(prefix)}\n\n{chunk_text}"
            
            # Create chunk document with metadata
            chunk_doc = Document(
                content=chunk_text,
                doc_id=f"{document.doc_id}_chunk_{len(chunks)}",
                metadata={
                    **document.metadata,
                    "chunk_index": len(chunks),
                    "original_doc_id": document.doc_id,
                    "title": title,
                    "section": current_section
                }
            )
            chunks.append(chunk_doc)
        
        return chunks

class PineconeStore:
    """Pinecone-based vector store for efficient similarity search."""
    
    def __init__(self, index_name: str = "default-index"):
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        self.pc = pinecone.Pinecone(api_key=self.api_key)
        self.index_name = index_name
        self.index = None
        self.initialize_index()

    def initialize_index(self, dimension: int = 3072):
        """Initialize or connect to Pinecone index."""
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

    def add_documents(self, documents: List[Document], embeddings: List[List[float]], namespace: str = "default"):
        """Add documents with their embeddings to Pinecone."""
        vectors_to_upsert = []
        
        for doc, embedding in tqdm(zip(documents, embeddings), desc="Adding documents to Pinecone"):
            vector_data = (
                doc.doc_id,
                embedding,
                {
                    "text": doc.content,
                    "title": doc.metadata.get("title", ""),
                    "section": doc.metadata.get("section", ""),
                    **{k: v for k, v in doc.metadata.items() 
                       if k not in ["title", "section"]}
                }
            )
            vectors_to_upsert.append(vector_data)
            
            # Batch upsert when we have enough vectors
            if len(vectors_to_upsert) >= 100:
                self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                vectors_to_upsert = []
        
        # Final batch upsert
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert, namespace=namespace)

    def search(self, query_embedding: List[float], top_k: int = 5, namespace: str = "default") -> List[Document]:
        """Search for similar documents using the query embedding."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True
        )
        
        documents = []
        for match in results.matches:
            # Reconstruct document with Wikipedia-specific metadata
            doc = Document(
                content=match.metadata["text"],
                doc_id=match.id,
                metadata={
                    "title": match.metadata.get("title", ""),
                    "section": match.metadata.get("section", ""),
                    "score": match.score,
                    **{k: v for k, v in match.metadata.items() 
                       if k not in ["text", "title", "section"]}
                }
            )
            documents.append(doc)
        
        return documents

    def delete_documents(self, doc_ids: List[str], namespace: str = "default"):
        """Delete documents from the index."""
        self.index.delete(ids=doc_ids, namespace=namespace)

class RAGDocumentStore:
    """Document store that combines document processing and vector storage for RAG."""
    
    def __init__(self, index_name: str = "default-index", chunk_size: int = 300, chunk_overlap: int = 120):
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = PineconeStore(index_name)
        
    def add_documents(self, documents: List[Document], embeddings: List[List[float]], namespace: str = "default"):
        """Process and add documents to the store."""
        processed_docs = []
        processed_embeddings = []
        
        # Process each document into chunks
        for doc, embedding in zip(documents, embeddings):
            chunks = self.processor.chunk_text(doc)
            processed_docs.extend(chunks)
            # Duplicate embedding for each chunk (simplified - in practice you'd want to re-embed each chunk)
            processed_embeddings.extend([embedding] * len(chunks))
        
        # Add to vector store
        self.vector_store.add_documents(processed_docs, processed_embeddings, namespace)
        
    def search(self, query_embedding: List[float], top_k: int = 5, namespace: str = "default") -> List[Document]:
        """Search for relevant documents."""
        return self.vector_store.search(query_embedding, top_k, namespace)
