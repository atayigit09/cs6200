from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models import BaseLLM
import torch
from typing import Dict, List, Any
import os

from models import BaseLLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig

from rag.document_store import RAGDocumentStore, Document, FaissVectorStore, ChromaVectorStore
from models import create_embedding_model

# from ollama import chat
# from ollama import ChatResponse

##baseline model
class BaselineLLaMA(BaseLLM):
    """
    Baseline LLaMA implementation using HF transformers with quantization
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.seed = config.get('seed')
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
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
        
        if self.seed is not None and 'seed' not in generation_config:
            generation_config['seed'] = self.seed

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

##rag model
class RagLLaMA(BaseLLM):
    """
    Retrieval-Augmented Generation model implementation.
    Extends the base LLM with retrieval capabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.seed = config.get('seed')
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.load_model()
        self.config = config
        self.load_document_store()
        self.initialize_embedding_model()
        
    
    def load_document_store(self):
        """Load the document store with vector database for a given field."""
        rag_config = self.config.get('rag', {})
        embedding_config = rag_config.get('embedding', {})
        vector_config = rag_config.get('vector_db', {})
        embedding_dim = embedding_config.get('embedding_dimension', 768)
        db_path = vector_config.get('db_path')
        vector_store_type = vector_config.get('type', 'faiss')
        field = rag_config.get('field')
        
        field_dir = os.path.join(db_path, field)
        
        if not os.path.exists(field_dir):
            raise FileNotFoundError(f"No embeddings found for field '{field}' at {field_dir}")
        
        if vector_store_type.lower() == 'chroma':
            vector_store = ChromaVectorStore(
                collection_name=field,
                persist_directory=field_dir
            )
            doc_store = RAGDocumentStore(vector_store=vector_store)
            print(f"Loaded Chroma vector store for field '{field}'")
        else:
            vector_store = FaissVectorStore(dimension=embedding_dim)
            vector_store = vector_store.load(os.path.join(field_dir, "vector_store"))
            doc_store = RAGDocumentStore(vector_store=vector_store)
            doc_store = doc_store.load(field_dir)
            print(f"Loaded FAISS vector store for field '{field}'")

            self.doc_store =  doc_store
    
    def initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        embedding_config = self.config.get("rag", {}).get("embedding", {})
        self.embedding_model = create_embedding_model(embedding_config)
    
    def load_model(self):
        model_config = self.config['model']
        quant_config = self.config.get('quantization', {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_id'],
            use_fast=model_config.get('use_fast', True)
        )

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

    def search(self, query: str) -> List[Document]:
        num_results = self.config.get("rag", {}).get("top_k", 5)
        """Search the document store for documents similar to the query."""
        query_embedding = self.embedding_model.embed([query])[0]
        results = self.doc_store.search(query_embedding, top_k=num_results)
        
        unique_results = []
        seen_content = set()
        
        for doc in results:
            if doc.content not in seen_content:
                seen_content.add(doc.content)
                unique_results.append(doc)
                
            if len(unique_results) >= num_results:
                break
        
        return unique_results[:num_results]
    
    
    def format_context(self, documents) -> str:
        rag_config = self.config.get("rag", {})
        context_format = rag_config.get("context_format", "simple")
        
        if context_format == "simple":
            context_parts = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", f"Document {i+1}")
                context_parts.append(f"Document {i+1} ({source}):\n{doc.content}\n")
            
            return "\n".join(context_parts)
        
        elif context_format == "compact":
            return "\n\n".join([doc.content for doc in documents])
        
        else:
            raise ValueError(f"Unknown context format: {context_format}")
    
    def format_prompt(self, query: str, context: str) -> str:
        rag_config = self.config.get("rag", {})
        prompt_template = rag_config.get("prompt_template", "default")
        
        if prompt_template == "default":
            return (
                "Below is an instruction that describes a task, paired with context information that might be relevant and could be used to gain more information. "
                "Write a response that appropriately completes the request.\n\n"
                "### Relevant Context:\n"
                f"{context}\n\n"
                "### Instruction:\n"
                f"{query}\n\n"
                "### Response:\n"
            )
        
        elif prompt_template == "concise":
            return (
                f"Context information is below.\n"
                f"---------------------\n"
                f"{context}\n"
                f"---------------------\n"
                f"Given the context information and not prior knowledge, answer the query.\n"
                f"Query: {query}\n"
                f"Answer: "
            )
        
        elif prompt_template == "custom":
            custom_template = rag_config.get("custom_prompt_template", "")
            if not custom_template:
                raise ValueError("Custom prompt template not provided")
            
            return custom_template.format(query=query, context=context)
        
        else:
            raise ValueError(f"Unknown prompt template: {prompt_template}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        documents = self.search(prompt)
        context = self.format_context(documents)

        if self.config['rag'].get('debug',False):
            print(f"Context: {context}")

        full_prompt = self.format_prompt(prompt, context)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        generation_config = {
            'max_new_tokens': self.config['generation']['max_length'],
            'temperature': self.config['generation']['temperature'],
            'top_p': self.config['generation']['top_p'],
            **kwargs
        }
        
        if self.seed is not None and 'seed' not in generation_config:
            generation_config['seed'] = self.seed

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        del inputs
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

#fine tune model
class LoraLLaMA(BaseLLM):
    """
    Unified LLaMA implementation that supports both QLoRA and standard LoRA fine tuning.

    The fine tuning mode is determined by the configuration under the "finetuning" key.
    Set either "use_qlora": true or "use_lora": true in your config. 
    
    The model can load from saved checkpoints under results/lora_model/checkpoint.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        model_config = self.config["model"]
        quant_config = self.config.get("quantization", {})
        # Check fine tuning configuration
        finetuning_config = self.config.get("finetuning", {})
        
        # Default checkpoint path
        checkpoint_path = finetuning_config.get("checkpoint_path", "results/lora_model/checkpoint")

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

        # Check if we should load from checkpoint
        if finetuning_config.get("load_from_checkpoint", False) and os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            
            # Load the base model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config["model_id"],
                **load_params
            )
            
            
            
            if finetuning_config.get("use_qlora", False):
                # QLoRA: Prepare the model for k-bit training before loading
                self.model = prepare_model_for_kbit_training(self.model)
                
            # Set up LoRA configuration for loading the adapter
            lora_config = LoraConfig(
                r=finetuning_config.get("lora_r", 8),
                lora_alpha=finetuning_config.get("lora_alpha", 32),
                target_modules=finetuning_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=finetuning_config.get("lora_dropout", 0.05),
                bias=finetuning_config.get("bias", "none"),
                task_type="CAUSAL_LM"
            )
            
            # Convert to PEFT model
            self.model = get_peft_model(self.model, lora_config)
            
            # Load adapter weights from checkpoint
            try:
                self.model.load_state_dict(torch.load(
                    os.path.join(checkpoint_path, "adapter_model.bin"),
                    map_location="cpu"
                ))
                print("Successfully loaded adapter weights from checkpoint")
            except Exception as e:
                print(f"Error loading adapter weights: {e}")
                
            self.print_trainable_parameters()
            
        else:
            # Load the base model without checkpoint
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config["model_id"],
                **load_params
            )

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


#rag ollama model
class RagOLLaMA(BaseLLM):
    """
    Retrieval-Augmented Generation model implementation.
    Extends the base LLM with retrieval capabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.seed = config.get('seed')
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set Python's random seed as well for the Ollama API calls
            import random
            random.seed(self.seed)
        
        self.config = config
        self.load_document_store()
        self.initialize_embedding_model()
        
    
    def load_document_store(self):
        """Load the document store with vector database for a given field."""
        rag_config = self.config.get('rag', {})
        embedding_config = rag_config.get('embedding', {})
        vector_config = rag_config.get('vector_db', {})
        embedding_dim = embedding_config.get('embedding_dimension', 768)
        db_path = vector_config.get('db_path')
        vector_store_type = vector_config.get('type', 'faiss')
        field = rag_config.get('field')
        
        field_dir = os.path.join(db_path, field)
        
        if not os.path.exists(field_dir):
            raise FileNotFoundError(f"No embeddings found for field '{field}' at {field_dir}")
        
        if vector_store_type.lower() == 'chroma':
            vector_store = ChromaVectorStore(
                collection_name=field,
                persist_directory=field_dir
            )
            doc_store = RAGDocumentStore(vector_store=vector_store)
            print(f"Loaded Chroma vector store for field '{field}'")
        else:
            vector_store = FaissVectorStore(dimension=embedding_dim)
            vector_store = vector_store.load(os.path.join(field_dir, "vector_store"))
            doc_store = RAGDocumentStore(vector_store=vector_store)
            doc_store = doc_store.load(field_dir)
            print(f"Loaded FAISS vector store for field '{field}'")

            self.doc_store =  doc_store
    
    def initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        embedding_config = self.config.get("rag", {}).get("embedding", {})
        self.embedding_model = create_embedding_model(embedding_config)
       

    def search(self, query: str) -> List[Document]:
        num_results = self.config.get("rag", {}).get("top_k", 5)
        """Search the document store for documents similar to the query."""
        query_embedding = self.embedding_model.embed([query])[0]
        results = self.doc_store.search(query_embedding, top_k=num_results)
        
        unique_results = []
        seen_content = set()
        
        for doc in results:
            if doc.content not in seen_content:
                seen_content.add(doc.content)
                unique_results.append(doc)
                
            if len(unique_results) >= num_results:
                break
        
        return unique_results[:num_results]
    
    
    def format_context(self, documents) -> str:
        rag_config = self.config.get("rag", {})
        context_format = rag_config.get("context_format", "simple")
        
        if context_format == "simple":
            context_parts = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", f"Document {i+1}")
                context_parts.append(f"Document {i+1} ({source}):\n{doc.content}\n")
            
            return "\n".join(context_parts)
        
        elif context_format == "compact":
            return "\n\n".join([doc.content for doc in documents])
        
        else:
            raise ValueError(f"Unknown context format: {context_format}")
    
    def format_prompt(self, query: str, context: str) -> str:
        rag_config = self.config.get("rag", {})
        prompt_template = rag_config.get("prompt_template", "default")
        
        if prompt_template == "default":
            return (
                "Below is an instruction that describes a task, paired with relevant context information. "
                "Write a response that appropriately completes the request.\n\n"
                "### Relevant Context:\n"
                f"{context}\n\n"
                "### Instruction:\n"
                f"{query}\n\n"
                "### Response:\n"
            )
        
        elif prompt_template == "concise":
            return (
                f"Context information is below.\n"
                f"---------------------\n"
                f"{context}\n"
                f"---------------------\n"
                f"Given the context information and not prior knowledge, answer the query.\n"
                f"Query: {query}\n"
                f"Answer: "
            )
        
        elif prompt_template == "custom":
            custom_template = rag_config.get("custom_prompt_template", "")
            if not custom_template:
                raise ValueError("Custom prompt template not provided")
            
            return custom_template.format(query=query, context=context)
        
        else:
            raise ValueError(f"Unknown prompt template: {prompt_template}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        documents = self.search(prompt)
        context = self.format_context(documents)

        if self.config['rag'].get('debug',False):
            print(f"Context: {context}")

        full_prompt = self.format_prompt(prompt, context)
        
        # For Ollama API calls, we need to set a seed parameter if available
        chat_params = {}
        if self.seed is not None:
            chat_params['seed'] = self.seed

        response: ChatResponse = chat(
            model='llama3.1:8b', 
            messages=[
                {
                    'role': 'user',
                    'content': full_prompt,
                },
            ],
            **chat_params
        )
                
        qas = response['message']['content']
        
        return qas

class FTLora(BaseLLM):
    """
    A class for loading fine-tuned LoRA models from checkpoints
    and generating text with them. This class focuses on inference rather than
    training.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.checkpoint_path = config['finetuning']['checkpoint_path']
        self.load_model()

    def load_model(self):
        """
        Load the base model and the fine-tuned LoRA adapter if a checkpoint path is provided.
        """
        model_config = self.config['model']
        quant_config = self.config.get('quantization', {})

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_id'],
            use_fast=model_config.get('use_fast', True)
        )
        
        # Set up padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        # Set up model loading parameters
        load_params = {
            "torch_dtype": torch.float16,
            "device_map": "cuda",
            "low_cpu_mem_usage": True
        }
        
        if bnb_config:
            load_params["quantization_config"] = bnb_config

        # Load the base model
        print(f"Loading base model from {model_config['model_id']}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            **load_params
        )
        
        # Load the fine-tuned adapter if a checkpoint path was provided
        if self.checkpoint_path:
            self.load_adapter(self.checkpoint_path)
            
    def load_adapter(self, checkpoint_path: str):
        """
        Load a fine-tuned LoRA adapter from a checkpoint.
        """
        try:
            print(f"Loading LoRA adapter from {checkpoint_path}...")
            # Load the PEFT configuration to verify it's a LoRA adapter
            peft_config = PeftConfig.from_pretrained(checkpoint_path)
            
            # Load the adapter weights
            self.model = PeftModel.from_pretrained(
                self.model,
                checkpoint_path,
                is_trainable=False  # Set to False for inference
            )
            
            print(f"Successfully loaded adapter from {checkpoint_path}")
            print(f"Adapter type: {peft_config.peft_type}")
            
            # Check if we should merge weights for faster inference
            if self.config.get('inference', {}).get('merge_weights', False):
                self.merge_lora_weights()
                
        except Exception as e:
            print(f"Error loading adapter checkpoint: {str(e)}")
            raise
    
    def merge_lora_weights(self):
        """
        Merge the LoRA adapter weights into the base model for faster inference.
        """
        if hasattr(self.model, "merge_and_unload"):
            print("Merging LoRA weights into base model...")
            self.model = self.model.merge_and_unload()
            print("LoRA weights merged successfully")
        else:
            print("Model doesn't have LoRA adapters to merge")
    
    def format_prompt(self, prompt: str) -> str:
        """
        Format the input prompt based on the template used during fine-tuning.
        This ensures consistent inputs between training and inference.
        """
        if self.config['generation'].get('format_prompt', False):
            return (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{prompt}\n\n"
                "### Response:\n"
            )
        return prompt
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response for the given prompt using the fine-tuned model.
        """
        # Format the prompt if needed
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize the input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        # Set up generation parameters
        generation_config = {
            'max_new_tokens': self.config['generation'].get('max_length', 512),
            'temperature': self.config['generation'].get('temperature', 0.7),
            'top_p': self.config['generation'].get('top_p', 0.9),
            'do_sample': self.config['generation'].get('do_sample', True),
            **kwargs
        }
        
        # Generate the response
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        # Clean up to save memory
        del inputs
        
        # Decode and return the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If we formatted the prompt, we need to remove the prompt from the response
        if self.config['generation'].get('format_prompt', False):
            # Find where the response starts and only return that part
            response_marker = "### Response:"
            if response_marker in response:
                response = response.split(response_marker)[-1].strip()
        
        return response