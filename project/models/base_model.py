from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models import BaseLLM
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }

        if bnb_config:
            load_params["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            **load_params
        )

    def format_prompt(self, prompt: str) -> str:
        """
        Formats the input prompt for an instruction-tuned LLaMA model.
        This template follows a structure that the model was fine-tuned on,
        helping it distinguish between the instruction and the expected response.
        """
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
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

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        del inputs
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_batches(self, prompts: list, **kwargs) -> list:
        # Optionally format each prompt if the configuration requires it.
        if self.config['generation'].get('format_prompt', False):
            prompts = [self.format_prompt(prompt) for prompt in prompts]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # or set a custom token

        # Tokenize the list of prompts as a batch.
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        # Prepare the generation configuration.
        generation_config = {
            'max_new_tokens': self.config['generation']['max_length'],
            'temperature': self.config['generation']['temperature'],
            'top_p': self.config['generation']['top_p'],
            **kwargs
        }

        # Generate outputs in batch without tracking gradients.
        with torch.no_grad():
            outputs_tensor = self.model.generate(**inputs, **generation_config)

        # Decode each output in the batch.
        outputs = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs_tensor
        ]

        del inputs  # Cleanup to free memory.
        
        return outputs



from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models import BaseLLM
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LoraLLaMA(BaseLLM):
    """
    Unified LLaMA implementation that supports both QLoRA and standard LoRA fine tuning.

    The fine tuning mode is determined by the configuration under the "finetuning" key.
    Set either "use_qlora": true or "use_lora": true in your config. 
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        model_config = self.config["model"]
        quant_config = self.config.get("quantization", {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_id"],
            use_fast=model_config.get("use_fast", True)
        )

        # Set up quantization configuration if specified.
        if quant_config.get("load_in_4bit", False) or quant_config.get("load_in_8bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.get("load_in_4bit", False),
                load_in_8bit=quant_config.get("load_in_8bit", False),
                bnb_4bit_compute_dtype=torch.bfloat16 if quant_config.get("load_in_4bit", False) else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        load_params = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        if bnb_config:
            load_params["quantization_config"] = bnb_config

        # Load the base model.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["model_id"],
            **load_params
        )

        # Check fine tuning configuration.
        finetuning_config = self.config.get("finetuning", {})

        if finetuning_config.get("use_qlora", False):
            # QLoRA: Prepare the model for k-bit training before applying LoRA.
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=finetuning_config.get("lora_r", 8),
                lora_alpha=finetuning_config.get("lora_alpha", 32),
                target_modules=finetuning_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=finetuning_config.get("lora_dropout", 0.05),
                bias=finetuning_config.get("bias", "none"),
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.print_trainable_parameters()

        elif finetuning_config.get("use_lora", False):
            # Standard LoRA without k-bit preparation.
            lora_config = LoraConfig(
                r=finetuning_config.get("lora_r", 8),
                lora_alpha=finetuning_config.get("lora_alpha", 32),
                target_modules=finetuning_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=finetuning_config.get("lora_dropout", 0.05),
                bias=finetuning_config.get("bias", "none"),
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """
        Prints the number and percentage of trainable parameters, verifying that only the
        LoRA (or QLoRA) parameters are being updated during fine tuning.
        """
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"Trainable params: {trainable_params} | All params: {all_params} "
            f"({100 * trainable_params / all_params:.2f}% trainable)"
        )
    
    def format_prompt(self, prompt: str) -> str:
        """
        Formats the input prompt using an instruction-response template.
        """
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generates a single output for the given prompt.
        """
        if self.config["generation"].get("format_prompt", False):
            prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generation_config = {
            "max_new_tokens": self.config["generation"]["max_length"],
            "temperature": self.config["generation"]["temperature"],
            "top_p": self.config["generation"]["top_p"],
            **kwargs
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        del inputs
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_batches(self, prompts: list, **kwargs) -> list:
        """
        Generates outputs for a batch of prompts.
        """
        if self.config["generation"].get("format_prompt", False):
            prompts = [self.format_prompt(prompt) for prompt in prompts]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generation_config = {
            "max_new_tokens": self.config["generation"]["max_length"],
            "temperature": self.config["generation"]["temperature"],
            "top_p": self.config["generation"]["top_p"],
            **kwargs
        }
        with torch.no_grad():
            outputs_tensor = self.model.generate(**inputs, **generation_config)

        outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs_tensor]
        del inputs
        return outputs
