import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from pymed import PubMed
from project.rag.scrappers import BaseScrapper
import time

class PubMedScraper(BaseScrapper):
    """
    A class for scraping PubMed articles based on keywords using the pymed library.
    """
    
    def __init__(self, field="Bio-Medical", docs_dir="project/data/docs", email=None, tool_name="RAGScraper"):
        """
        Initialize the PubMed scraper.
        
        Args:
            field (str): Field/subfolder to save documents in
            docs_dir (str): Base directory for storing documents
            email (str): Email to use with the NCBI API (required by NCBI)
            tool_name (str): Name of your tool for NCBI tracking
        """
        self.docs_dir = Path(docs_dir + "/" + field + "/PubMed")
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Email is required by NCBI for API usage
        if email is None:
            raise ValueError("Email is required for PubMed API access")
            
        self.pubmed = PubMed(tool=tool_name, email=email)
    
    def search_pubmed(self, keyword: str, max_results: int = 5) -> List[Any]:
        """
        Search PubMed for articles matching a keyword.
        
        Args:
            keyword (str): The keyword to search for
            max_results (int): Maximum number of results to return
            
        Returns:
            List[Any]: List of PubMed article objects
        """
        try:
            # Query PubMed with the keyword
            results = self.pubmed.query(keyword, max_results=max_results)
            # Convert results iterator to a list
            articles = list(results)
            time.sleep(2)
            return articles
        
        except Exception as e:
            print(f"Error searching PubMed for '{keyword}': {str(e)}")
            return []
    
    def extract_article_data(self, article) -> Dict[str, Any]:
        """
        Extract relevant data from a pymed article object.
        
        Args:
            article: PubMed article object from pymed
            
        Returns:
            Dict[str, Any]: Structured article data
        """
        # Extract basic article information
        article_data = {
            "title": article.title if hasattr(article, 'title') else "No Title",
            "abstract": article.abstract if hasattr(article, 'abstract') else "No abstract available",
            "journal": article.journal if hasattr(article, 'journal') else "Unknown Journal",
            "publication_date": str(article.publication_date) if hasattr(article, 'publication_date') else "Unknown Date",
            "pmid": article.pubmed_id if hasattr(article, 'pubmed_id') else "No PMID",
            "doi": article.doi if hasattr(article, 'doi') else "No DOI",
            "authors": [],
            "keywords": [],
            "mesh_terms": []
        }
        
        # Extract authors
        if hasattr(article, 'authors') and article.authors:
            article_data["authors"] = [author.get('name', '') for author in article.authors]
        
        # Extract keywords
        if hasattr(article, 'keywords') and article.keywords:
            article_data["keywords"] = article.keywords
        
        # Extract mesh terms
        if hasattr(article, 'mesh_terms') and article.mesh_terms:
            article_data["mesh_terms"] = article.mesh_terms
        
        return article_data
    
    def clean_filename(self, title: str) -> str:
        """
        Create a safe filename from an article title.
        
        Args:
            title (str): The article title
            
        Returns:
            str: A sanitized filename
        """
        # Replace spaces with underscores and remove invalid characters
        safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        # Ensure filename is not too long
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name
    
    def process_content(self, article_data: Dict[str, Any]) -> str:
        """
        Format article data into a structured document.
        
        Args:
            article_data (Dict[str, Any]): Article information
            
        Returns:
            str: Formatted article content
        """
        # Format authors
        authors_str = ", ".join(article_data["authors"]) if article_data["authors"] else "Unknown"
        
        # Format keywords
        keywords = "\n- " + "\n- ".join(article_data["keywords"]) if article_data["keywords"] else "None"
        
        # Format MeSH terms
        mesh_terms = "\n- " + "\n- ".join(article_data["mesh_terms"]) if article_data["mesh_terms"] else "None"
        
        # Build formatted content
        content = f"""# {article_data['title']}

## Metadata
- **Authors**: {authors_str}
- **Journal**: {article_data['journal']}
- **Publication Date**: {article_data['publication_date']}
- **PMID**: {article_data['pmid']}
- **DOI**: {article_data['doi']}

## Abstract
{article_data['abstract']}

## Keywords
{keywords}

## MeSH Terms
{mesh_terms}
"""
        return content
    
    def fetch_and_save(self, keyword: str, use_exact_keyword_as_filename: bool = True) -> Optional[str]:
        """
        Search for, fetch, and save PubMed articles based on a keyword.
        
        Args:
            keyword (str): The keyword to search for
            use_exact_keyword_as_filename (bool): Whether to use the keyword or article title for filename
            
        Returns:
            Optional[str]: Path to the saved file, or None if operation failed
        """
        # Search for articles matching the keyword
        articles = self.search_pubmed(keyword)
        
        if not articles:
            print(f"No PubMed results found for '{keyword}'")
            return None
        
        # Use the first (most relevant) search result
        first_article = articles[0]
        article_data = self.extract_article_data(first_article)
        
        # Determine filename
        if use_exact_keyword_as_filename:
            filename = self.clean_filename(keyword) + ".txt"
        else:
            filename = self.clean_filename(article_data["title"]) + ".txt"
        
        file_path = self.docs_dir / filename
        
        # Format the content
        formatted_content = self.process_content(article_data)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            print(f"Saved PubMed article to {file_path}")
            return str(file_path)
        
        except Exception as e:
            print(f"Error saving article for '{keyword}': {str(e)}")
            return None
    
    def fetch_multiple(self, keywords: List[str], max_per_keyword: int = 3) -> List[str]:
        """
        Fetch multiple articles for a list of keywords.
        
        Args:
            keywords (List[str]): List of keywords to search for
            max_per_keyword (int): Maximum number of articles to fetch per keyword
            
        Returns:
            List[str]: List of paths to saved files
        """
        saved_files = []
        
        for keyword in keywords:
            print(f"Processing keyword: {keyword}")
            articles = self.search_pubmed(keyword, max_results=max_per_keyword)
            
            for i, article in enumerate(articles):
                article_data = self.extract_article_data(article)
                
                # Create a filename that includes the keyword and article number
                filename = f"{self.clean_filename(keyword)}_article_{i+1}.txt"
                file_path = self.docs_dir / filename
                
                # Format and save the content
                formatted_content = self.process_content(article_data)
                
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(formatted_content)
                    
                    saved_files.append(str(file_path))
                    print(f"Saved PubMed article to {file_path}")
                
                except Exception as e:
                    print(f"Error saving article for article {i+1} of keyword '{keyword}': {str(e)}")
        
        return saved_files