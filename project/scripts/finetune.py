import argparse
import yaml
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from models import create_model


def load_model_config():
    """Loads the configuration file for the model."""
    config_path = Path("configs/base_model.yaml").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found!")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def format_instruction(example):
    """Format examples in the instruction-tuning format."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    response = example["output"]
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    example["text"] = prompt + response
    return example


def preprocess_data(dataset_path, tokenizer, max_length):
    """Load and preprocess the dataset for training."""
    print(f"Loading dataset from {dataset_path}")
    
    # Load the dataset
    dataset = load_dataset("json", data_files=dataset_path)
    
    # Format the dataset into instruction-tuning format
    formatted_dataset = dataset["train"].map(format_instruction)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"]
    )
    
    return tokenized_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with LoRA")
    
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the instruction tuning dataset (JSON format)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directory to save fine-tuned model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    args.model_config = load_model_config()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    print("Creating LoraLLaMA model...")
    # Make sure 'finetuning' section exists in config and has 'use_lora' or 'use_qlora' set to True
    if 'finetuning' not in args.model_config:
        raise ValueError("No 'finetuning' section in config file")
    
    if not (args.model_config['finetuning'].get('use_lora', False) or 
            args.model_config['finetuning'].get('use_qlora', False)):
        raise ValueError("Neither 'use_lora' nor 'use_qlora' is set to True in config")
    
    model = create_model(args, model_class='LoraLLaMA')
    tokenizer = model.tokenizer
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading and preprocessing dataset from {args.dataset_path}...")
    train_dataset = preprocess_data(args.dataset_path, tokenizer, args.max_seq_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,  # Enable mixed precision training
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_total_limit=3,  # Only keep the 3 most recent checkpoints
        push_to_hub=False,
        report_to="none",  # Disable reporting to integrations like Weights & Biases
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save model
    final_checkpoint_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    trainer.save_model(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    
    print(f"Training complete. Model saved to {final_checkpoint_dir}") 