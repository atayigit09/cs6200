from typing import Union, List, Dict, Any
import os
import torch
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from anthropic import Anthropic


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


class EvalLLM(BaseLLM):
    """
    EvalLLM for extracting and validating factual statements from model responses.
    Supports multiple LLM backends (GPT, Claude) for fact checking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EvalLLM with configuration
        
        Args:
            config: Dictionary containing:
                - model_type: "gpt" or "claude"
                - api_key: API key for the chosen service
                - model_name: Specific model name (e.g. "gpt-4o", "claude-3.5-sonnet")
                - data_path: Path to input data
                - save_path: Path to save results
                - batch_size: Number of items to process before saving
        """
        super().__init__(config)
        self.model_type = config['model_type']
        self.model_name = config['model_name']
        self.data_path = config['data_path']
        self.save_path = config['save_path']
        self.batch_size = config.get('batch_size', 32)
        self.save_data = []

        # Initialize appropriate client based on model type
        if self.model_type == "gpt":
            openai.api_key = config['api_key']
            self.client = openai.OpenAI()
        elif self.model_type == "claude":
            config_key = config.get('api_key')
            self.client = Anthropic(api_key=config_key)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    async def generate_completion(self, prompt: str) -> str:
        """Generate completion using the configured LLM"""
        try:
            if self.model_type == "gpt":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif self.model_type == "claude":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.0,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    system="You are a helpful AI assistant focused on fact checking and verification."
                )
                return response.content[0].text
                
        except Exception as e:
            print(f"Error generating completion: {str(e)}")
            return ""

    def get_facts_lst(self, response: str) -> List[str]:
        """Extract facts list from the LLM response."""
        if not response or "NO FACTS" in response:
            return []
        
        try:
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            
            if not lines:
                print(f"Empty facts: {response}")
                return []
                
            if len(lines) == 1 and not lines[0].startswith("1."):
                return [lines[0]]
                
            return [fact[2:].strip() for fact in lines if fact[2:].strip()]
            
        except Exception as e:
            print(f"Error parsing facts: {str(e)}")
            print(f"Response: {response}")
            return []

    async def generate_facts(self, data: List[Dict], prompt_template: str) -> List[Dict]:
        """
        Generate facts from model responses using the configured LLM.
        
        Args:
            data: List of dictionaries containing model responses
            prompt_template: Template for fact extraction prompt
        
        Returns:
            Updated data with extracted facts
        """
        if not data:
            return []

        for i in tqdm(range(len(data)), desc="Extracting facts"):
            try:
                # Format prompt with current item
                prompt = prompt_template.format(
                    query=data[i]["user_query"],
                    answer=data[i]["model_response"]
                )

                # Generate and process facts
                response = await self.generate_completion(prompt)
                facts = self.get_facts_lst(response)
                
                # Update item with results
                data[i].update({
                    "raw_facts": response,
                    "extracted_facts": facts
                })
                
                self.save_data.append(data[i])
                
                # Save periodically
                if len(self.save_data) % self.batch_size == 0:
                    self.save_results()
                    
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                continue

        # Final save
        self.save_results()
        return data

    def save_results(self):
        """Save current results to disk"""
        if not self.save_data:
            return
            
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(self.save_data, f, indent=2)