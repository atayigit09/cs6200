from typing import Dict, List, Any, Union
import importlib

#inherit this class in all scrapper classes so that all the scrapper classes have the same interface!!!

class BaseScrapper:
    """Base class for all scrappers"""
    def __init__(self):
        pass
        
    def fetch_and_save(self):
        """Common model loading logic"""
        raise NotImplementedError
        