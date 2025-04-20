from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from project.models import BaseLLM
import torch
import os
import dotenv

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from project.rag.document_store import RAGDocumentStore, Document, FaissVectorStore, ChromaVectorStore
from project.models import create_embedding_model


##baseline model
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

        # Load HF token from .env file
        dotenv.load_dotenv()
        hf_token = os.getenv("token")
        if hf_token:
            print("Using Hugging Face token from .env file")
        else:
            print("WARNING: No Hugging Face token found in .env file")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['model_id'],
            use_fast=model_config.get('use_fast', True),
            token=hf_token
        )

        # Print quantization settings for debugging
        print(f"Quantization settings:")
        print(f"  load_in_4bit: {quant_config.get('load_in_4bit', False)}")
        print(f"  load_in_8bit: {quant_config.get('load_in_8bit', False)}")

        # Define correct quantization settings
        if quant_config.get('load_in_4bit', False) or quant_config.get('load_in_8bit', False):
            print("Applying quantization...")
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.get('load_in_4bit', False),
                load_in_8bit=quant_config.get('load_in_8bit', False),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print(f"BitsAndBytesConfig: {bnb_config}")
        else:
            bnb_config = None
            print("No quantization being applied")

        load_params = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "token": hf_token
        }

        if bnb_config:
            load_params["quantization_config"] = bnb_config

        print(f"Loading model with params: {load_params}")
        
        # Ensure required packages are available
        try:
            import bitsandbytes
            print(f"bitsandbytes version: {bitsandbytes.__version__}")
        except ImportError:
            print("WARNING: bitsandbytes not installed! Quantization may not work correctly.")
            print("Install with: pip install bitsandbytes")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_config['model_id'],
            **load_params
        )
        
        # Verify quantization was applied correctly
        if hasattr(self.model, 'is_quantized'):
            print(f"Model quantization status: {self.model.is_quantized}")
        
        # Check model size in memory
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded with {total_params/1_000_000:.2f}M parameters")
        
        # Check if model is actually quantized by examining the data type of weights
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                print(f"Module {name}: weight dtype = {module.weight.dtype}")
                break  # Just print the first one as an example

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

##rag model
class RagLLaMA(BaseLLM):
    """
    Retrieval-Augmented Generation model implementation.
    Extends the base LLM with retrieval capabilities.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize components
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
            # For Chroma, we load the collection
            vector_store = ChromaVectorStore(
                collection_name=field,
                persist_directory=field_dir
            )
            doc_store = RAGDocumentStore(vector_store=vector_store)
            print(f"Loaded Chroma vector store for field '{field}'")
        else:
            # For FAISS, we load the index
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

    def search(self, query: str) -> List[Document]:
        num_results = self.config.get("rag", {}).get("top_k", 5)
        """Search the document store for documents similar to the query."""
        # Embed the query
        query_embedding = self.embedding_model.embed([query])[0]
        # Search for similar documents
        results = self.doc_store.search(query_embedding, top_k=num_results)  # Retrieve more than needed to handle duplicates
        
        # Deduplicate results by content
        unique_results = []
        seen_content = set()
        
        for doc in results:
            if doc.content not in seen_content:
                seen_content.add(doc.content)
                unique_results.append(doc)
                
            # Stop once we have enough unique documents
            if len(unique_results) >= num_results:
                break
        
        return unique_results[:num_results]
    
    
    def format_context(self, documents) -> str:

        rag_config = self.config.get("rag", {})
        context_format = rag_config.get("context_format", "simple")
        
        if context_format == "simple":
            # Simple concatenation with document separators
            context_parts = []
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", f"Document {i+1}")
                context_parts.append(f"Document {i+1} ({source}):\n{doc.content}\n")
            
            return "\n".join(context_parts)
        
        elif context_format == "compact":
            # More compact format
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

        # Format context and prompt
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
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        model_config = self.config["model"]
        quant_config = self.config.get("quantization", {})
        finetuning_config = self.config.get("finetuning", {})
        
        # Check if we should load from a checkpoint
        lora_checkpoint = model_config.get("lora_checkpoint", None)

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

        # If we're loading from a checkpoint, load the fine-tuned model
        if lora_checkpoint:
            print(f"Loading LoRA weights from checkpoint: {lora_checkpoint}")
            if os.path.exists(lora_checkpoint):
                # Load the fine-tuned model using PEFT
                self.model = PeftModel.from_pretrained(
                    self.model,
                    lora_checkpoint,
                    is_trainable=False  # Set to True if we want to continue training
                )
                print(f"Successfully loaded LoRA weights from {lora_checkpoint}")
            else:
                print(f"Warning: LoRA checkpoint {lora_checkpoint} not found, using base model")
        # If no checkpoint, configure for training if needed
        elif finetuning_config.get("use_qlora", False):
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
