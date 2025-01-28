from typing import Union
import os
import torch
import json
import argparse
from tqdm import tqdm
from typing import Dict, Any
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseLLM:
    """Base class for all LLM models"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Common model loading logic"""
        raise NotImplementedError
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Common generation interface"""
        raise NotImplementedError

class BasePipeline:
    """Common pipeline for data handling"""
    def __init__(self, data_path: str, save_path: str):
        self.data_path = data_path
        self.save_path = save_path
        self.save_data = []
        
    def load_data(self) -> list:
        with open(self.data_path, 'r') as f:
            return json.load(f)
            
    def save_results(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.save_data, f, indent=2)

