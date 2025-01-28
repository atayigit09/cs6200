from models import BaseLLM, BasePipeline
from typing import Dict, Any
from tqdm import tqdm
import json

class HallucinationEvalPipeline(BasePipeline):
    """Evaluation pipeline for hallucination analysis"""
    def __init__(self, model: BaseLLM, data_path: str, save_path: str):
        super().__init__(data_path, save_path)
        self.model = model
        self.batch_size = 32
        
    def format_prompt(self, question: str) -> str:
        """LLaMA-2 specific prompt formatting"""
        return f"[INST] <<SYS>>\nYou are a helpful, factual assistant. Answer concisely.\n<</SYS>>\n\n{question} [/INST]"
    
    def run_evaluation(self):
        data = self.load_data()
        
        for item in tqdm(data, desc="Evaluating"):
            try:
                formatted_prompt = self.format_prompt(item['question'])
                response = self.model.generate(formatted_prompt)
                self.save_data.append({
                    'id': item['id'],
                    'question': item['question'],
                    'response': response
                })
                
                if len(self.save_data) % self.batch_size == 0:
                    self.save_results()
                    
            except Exception as e:
                print(f"Error processing {item['id']}: {str(e)}")
                
        self.save_results()