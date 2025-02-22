from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models import BaseLLM
import torch


class BaselineLLaMA(BaseLLM):
    """
    Baseline LLaMA implementation using HF transformers with quantization
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        model_config = self.config['model']
        quant_config = self.config.get('quantization', {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_id'],
            use_fast=model_config.get('use_fast', True)
        )

        # Define correct quantization settings
        if quant_config.get('load_in_4bit', False) or quant_config.get('load_in_8bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.get('load_in_4bit', False),
                load_in_8bit=quant_config.get('load_in_8bit', False),
                bnb_4bit_compute_dtype=torch.bfloat16 if quant_config.get('load_in_4bit', False) else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        load_params = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }

        if bnb_config:
            load_params["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            **load_params
        )

    def format_prompt(self, question: str) -> str:
        """LLaMA-3 specific prompt formatting"""
        return (
            "[INST] <<SYS>>\n"
            "You are a highly advanced, factual assistant powered by LLaMA-3. "
            "Provide clear, concise, and accurate responses.\n"
            "<</SYS>>\n\n"
            f"{question} [/INST]"
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        prompt = self.format_prompt(prompt) if self.config['generation']['format_prompt'] else prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        generation_config = {
            'max_new_tokens': self.config['generation']['max_length'],
            'temperature': self.config['generation']['temperature'],
            'top_p': self.config['generation']['top_p'],
            **kwargs
        }

        outputs = self.model.generate(**inputs, **generation_config)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
