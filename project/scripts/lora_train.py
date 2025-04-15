import os
import json
import torch
import argparse
import yaml
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset as HFDataset

# Import the model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import LoraLLaMA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        # Process all QA pairs
        for item in tqdm(data, desc="Processing dataset"):
            question = item["question"]
            answer = item["answer"]
            
            # Format the instruction using the same template as in LoraLLaMA class
            prompt = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n"
                f"{question}\n\n"
                "### Response:\n"
                f"{answer}"
            )
            
            self.inputs.append(prompt)
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

def load_data(data_path):
    """Load QA data from a JSON file."""
    with open(data_path, 'r') as f:
        return json.load(f)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples and prepare them for training."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    result["labels"] = result["input_ids"].clone()
    return result

def update_config_for_training(config, args):
    """Update configuration with training-specific settings."""
    # Ensure finetuning is enabled (either LoRA or QLoRA)
    if not config.get('finetuning', {}).get('use_lora', False) and not config.get('finetuning', {}).get('use_qlora', False):
        logger.warning("Neither LoRA nor QLoRA is enabled in the config. Enabling LoRA by default.")
        if 'finetuning' not in config:
            config['finetuning'] = {}
        config['finetuning']['use_lora'] = True
    
    
    # Fix quotation marks in finetuning parameters if they exist
    if 'finetuning' in config:
        for key in list(config['finetuning'].keys()):
            if key.endswith('"'):
                # Remove quotes from key names
                clean_key = key.rstrip('"')
                config['finetuning'][clean_key] = config['finetuning'].pop(key)
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LoraLLaMA model on QA dataset")
    parser.add_argument("--field", choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                        required=True, help="Select the topic to use for fine-tuning")
    parser.add_argument("--config", default="project/configs/base_model.yaml",
                       help="Path to the model configuration file")
    parser.add_argument("--output-dir", default="project/results/lora_model",
                       help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate for training")
    parser.add_argument("--max-length", type=int, default=1024,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config for training
    config = update_config_for_training(config, args)
    
    # Initialize the model
    logger.info("Initializing the LoraLLaMA model...")
    model = LoraLLaMA(config)

    tokenizer = model.tokenizer
    
    # If tokenizer doesn't have pad_token, set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    data_path = f"project/data/fineTune/{args.field}/{args.field}.json"
    logger.info(f"Loading data from {data_path}...")
    data = load_data(data_path)
    
    # Create dataset
    qa_dataset = QADataset(data, tokenizer, max_length=args.max_length)
    
    # Convert to Hugging Face dataset format
    hf_dataset = HFDataset.from_dict({"text": qa_dataset.inputs})
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = hf_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=["text"]
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main() 