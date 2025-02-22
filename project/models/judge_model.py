from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

from models import BaseLLM


class JudgeModel(BaseLLM):
    """
    Judge model implementation with HF
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.load_model()

    def load_model(self):
        model_config = self.config['model']

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['path'],
            use_fast=model_config.get('use_fast', True)
        )

        load_params = {
            'trust_remote_code': True
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['path'],
            **load_params
        )

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_config = {
            'max_length': self.config['generation']['max_length'],
            'temperature': self.config['generation']['temperature'],
            'top_p': self.config['generation']['top_p'],
            **kwargs
        }

        outputs = self.model.generate(
            **inputs,
            **generation_config
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)