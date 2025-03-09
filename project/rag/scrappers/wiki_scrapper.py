import re
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

class WikipediaScraper:
    """
    A class for scraping Wikipedia articles based on keywords.
    """
    
    def __init__(self, field = "Science", docs_dir: str = "data/docs"):
        self.docs_dir = Path(docs_dir + "/" + field)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://en.wikipedia.org/w/api.php"
    
    def search_wikipedia(self, keyword: str, limit: int = 5) -> List[Dict[str, str]]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": keyword,
            "srlimit": limit,
            "utf8": 1
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "query" in data and "search" in data["query"]:
                return [{"title": item["title"], "pageid": item["pageid"]} for item in data["query"]["search"]]
            return []
        
        except Exception as e:
            print(f"Error searching Wikipedia for '{keyword}': {str(e)}")
            return []
    
    def get_article_content(self, page_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|categories",
            "pageids": page_id,
            "explaintext": 1,  # Get plain text content
            "exsectionformat": "plain"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "query" in data and "pages" in data["query"]:
                page_data = next(iter(data["query"]["pages"].values()))
                return {
                    "title": page_data.get("title", ""),
                    "content": page_data.get("extract", ""),
                    "categories": [cat["title"] for cat in page_data.get("categories", [])]
                }
            return None
        
        except Exception as e:
            print(f"Error fetching Wikipedia article content for page ID {page_id}: {str(e)}")
            return None
    
    def clean_filename(self, title: str) -> str:
        # Replace spaces with underscores and remove invalid characters
        safe_name = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        return safe_name
    
    def fetch_and_save(self, keyword: str, use_exact_keyword_as_filename: bool = True) -> Optional[str]:
        # Search for articles matching the keyword
        search_results = self.search_wikipedia(keyword)
        
        if not search_results:
            print(f"No Wikipedia results found for '{keyword}'")
            return None
        
        # Use the first (most relevant) search result
        first_result = search_results[0]
        article_data = self.get_article_content(first_result["pageid"])
        
        if not article_data or not article_data["content"]:
            print(f"Could not fetch content for '{keyword}'")
            return None
        
        # Determine filename
        if use_exact_keyword_as_filename:
            filename = self.clean_filename(keyword) + ".txt"
        else:
            filename = self.clean_filename(article_data["title"]) + ".txt"
        
        file_path = self.docs_dir / filename

        # process the content
        article_data['content'] = article_data['content'].split("See also")[0]

        # Format and save the content
        formatted_content = f"# {article_data['title']}\n\n{article_data['content']}"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            print(f"Saved Wikipedia article to {file_path}")
            return str(file_path)
        
        except Exception as e:
            print(f"Error saving article for '{keyword}': {str(e)}")
            return None


