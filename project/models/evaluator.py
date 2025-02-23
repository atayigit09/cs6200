# coding: utf-8
from models import BaseLLM
from typing import  Dict, Any, List
import openai
from anthropic import Anthropic

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
                - provider: "openai" or "claude"
                - api_key: API key for the chosen service
                - model_name: Specific model name (e.g. "gpt-4o", "claude-3.5-sonnet")
        """
        super().__init__(config)
        self.load_model()

    
    def load_model(self):
        self.provider = self.config['provider']
        self.model_name = self.config['model_name']
        # Initialize appropriate client based on model type
        if self.provider == "openai":
            openai.api_key = self.config['api_key']
            # Instead of creating a new client, just assign the module as your client
            self.client = openai
        elif self.provider == "claude":
            config_key = self.config.get('api_key')
            self.client = Anthropic(api_key=config_key)
        else:
            raise ValueError(f"Unsupported model type: {self.provider}")


    def generate(self, prompt: str) -> str:
        """Generate completion using the configured LLM"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                            {"role": "system", "content": '''You are a helpful AI assistant focused on fact checking and verification.'''},
                            {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.get("temperature", 0.0),
                    max_tokens=self.config.get("max_tokens", 1000)
                )
                return response.choices[0].message.content
            
            elif self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.config.get("max_tokens", 1000),
                    temperature=self.config.get("temperature", 0.0),
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


