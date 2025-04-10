import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import random


def convert_format(questions: List[Dict[str, Any]], ground_truth_file=None) -> List[Dict[str, Any]]:
    """
    Convert HaluEval data from our format to a format suitable for instruction fine-tuning.
    Format needed: {"instruction": str, "input": str, "output": str}
    
    Args:
        questions: List of question dictionaries from HaluEval2
        ground_truth_file: Optional path to a ground truth file with verified answers
        
    Returns:
        List of formatted examples for instruction tuning
    """
    # If ground truth file is provided, load it
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    
    formatted_data = []
    
    for question in questions:
        # Extract query
        query = question.get("user_query", "")
        question_id = question.get("id", "")
        
        # Look for ground truth answer if available
        answer = ""
        if str(question_id) in ground_truth:
            # Use the ground truth answer if available
            answer = ground_truth[str(question_id)]
        elif "expert_answers" in question:
            # Use expert answers if available (assuming these are high-quality)
            answer = question["expert_answers"]
        elif "gpt4_answers" in question:
            # Use GPT-4 answers if available (assuming these are high-quality)
            answer = question["gpt4_answers"]
        
        # Skip if no answer is available
        if not answer:
            continue
            
        # Create the formatted example 
        formatted_example = {
            "instruction": query,
            "input": "",  # No separate input in our case
            "output": answer
        }
        
        formatted_data.append(formatted_example)
    
    return formatted_data


def main():
    """Parse arguments and prepare the fine-tuning dataset."""
    parser = argparse.ArgumentParser(description="Prepare HaluEval2 data for fine-tuning")
    
    parser.add_argument("--input_dir", type=str, default="data/HaluEval2",
                        help="Directory containing HaluEval2 data")
    parser.add_argument("--output_file", type=str, default="data/finetune_data.json",
                        help="Output path for prepared dataset")
    parser.add_argument("--ground_truth", type=str, default=None,
                        help="Path to ground truth data (if available)")
    parser.add_argument("--fields", type=str, nargs='+', 
                        default=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science"],
                        help="Fields to include in the dataset")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Portion of data to use as test set (0-1)")
    
    args = parser.parse_args()
    
    # Get all data files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    # Load and combine all questions
    all_questions = []
    for field in args.fields:
        field_file = input_dir / f"{field}.json"
        if not field_file.exists():
            print(f"Warning: Field file {field_file} not found, skipping")
            continue
            
        with open(field_file, 'r') as f:
            questions = json.load(f)
            print(f"Loaded {len(questions)} questions from {field_file}")
            all_questions.extend(questions)
    
    print(f"Total questions loaded: {len(all_questions)}")
    
    # Convert to fine-tuning format
    formatted_data = convert_format(all_questions, args.ground_truth)
    print(f"Formatted {len(formatted_data)} examples for fine-tuning")
    
    # Split into train and test if needed
    if args.test_split > 0:
        random.shuffle(formatted_data)
        test_size = int(len(formatted_data) * args.test_split)
        train_data = formatted_data[test_size:]
        test_data = formatted_data[:test_size]
        
        # Write train data
        train_path = Path(args.output_file)
        os.makedirs(train_path.parent, exist_ok=True)
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Wrote {len(train_data)} examples to {train_path}")
        
        # Write test data
        test_path = train_path.parent / f"test_{train_path.name}"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"Wrote {len(test_data)} examples to {test_path}")
    else:
        # Write all data to output file
        output_path = Path(args.output_file)
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        print(f"Wrote {len(formatted_data)} examples to {output_path}")


if __name__ == "__main__":
    main() 