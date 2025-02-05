from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import BaseLLM
import torch


class BaselineLLaMA(BaseLLM):
    """
    Baseline LLaMA implementation using HF transformers
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        model_config = self.config['model']
        quant_config = self.config['quantization']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['path'],
            use_fast=model_config.get('use_fast', True)
        )
        
        load_params = {
            'device_map': 'auto',
            'low_cpu_mem_usage': True
        }
        
        if quant_config['load_in_4bit']:
            load_params.update({
                'load_in_4bit': True,
                'bnb_4bit_compute_dtype': torch.bfloat16
            })
        elif quant_config['load_in_8bit']:
            load_params['load_in_8bit'] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['path'],
            **load_params
        )
        
    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = {
            'max_new_tokens': self.config['generation']['max_length'],
            'temperature': self.config['generation']['temperature'],
            'top_p': self.config['generation']['top_p'],
            **kwargs
        }
        
        outputs = self.model.generate(
            **inputs,
            **generation_config
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
