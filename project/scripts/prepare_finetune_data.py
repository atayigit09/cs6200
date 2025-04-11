import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Dict
import yaml

# Add parent directory to sys.path to allow imports from project module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import path utilities
from project.utils.paths import CONFIGS_DIR, DATA_DIR

# Import model creation functions
from project.models import create_model

def load_model_config():
    """Loads the configuration file for the model."""
    # Try multiple locations for config file
    config_path = CONFIGS_DIR / "base_model.yaml"
    project_config_path = Path(__file__).parent.parent / "configs" / "base_model.yaml"
    
    if config_path.exists():
        config_file = config_path
    elif project_config_path.exists():
        config_file = project_config_path
    else:
        raise FileNotFoundError(f"Config file not found at {config_path} or {project_config_path}")

    print(f"Loading config from: {config_file}")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    return config


def split_into_chunks(text: str, chunk_size: int = 4000, overlap: int = 500) -> List[str]:
    """
    Split a document into overlapping chunks to maintain context between chunks.
    
    Args:
        text: The document text to split
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk of text with specified size
        end = start + chunk_size
        
        # If not at the end of text, try to find a good break point
        if end < len(text):
            # Look for paragraph or sentence break near the end of the chunk
            paragraph_break = text.rfind('\n\n', start, end)
            sentence_break = text.rfind('. ', start, end)
            
            # Use paragraph break if found and it's not too far back
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2  # Include the paragraph break
            # Otherwise use sentence break if found
            elif sentence_break != -1 and sentence_break > start + chunk_size // 2:
                end = sentence_break + 2  # Include the period and space
        
        # Get the chunk
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position for next chunk, including overlap
        start = end - overlap
        
        # Ensure we're not stuck in an infinite loop
        if start >= len(text) or (len(chunks) > 1 and start >= len(text) - overlap):
            break
    
    return chunks


def extract_questions_from_document_with_llama(model, document_text: str, num_questions: int = 3, 
                                             chunk_size: int = 4000, chunk_overlap: int = 500) -> List[Dict[str, str]]:
    """
    Generate questions from a document using the BaselineLLaMA model.
    
    Args:
        model: The LLaMA model instance
        document_text: The text content of the document
        num_questions: Number of questions to generate PER DOCUMENT (not per chunk)
        chunk_size: Maximum size of document chunks in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of question-answer pairs in dict format, limited to num_questions
    """
    # Split document into chunks if it's too long
    chunks = split_into_chunks(document_text, chunk_size=chunk_size, overlap=chunk_overlap)
    all_qa_pairs = []
    
    print(f"Document split into {len(chunks)} chunks. Processing each chunk...")
    
    # Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}, length: {len(chunk)} chars")
        
        # Use the generate_ft_data method to get question-answer pairs
        result = model.generate_ft_data(chunk)
        print(f"Generated QA pairs for chunk {i+1}:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        # Try different approaches to extract valid JSON from the result
        qa_pairs = []
        
        # Approach 1: Try direct JSON parsing
        try:
            qa_pairs = json.loads(result)
            print(f"Successfully parsed full JSON response for chunk {i+1}")
        except json.JSONDecodeError:
            # Approach 2: Try to extract JSON array using regex
            try:
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result, re.DOTALL)
                if json_match:
                    qa_pairs = json.loads(json_match.group(0))
                    print(f"Successfully extracted JSON array using regex for chunk {i+1}")
            except json.JSONDecodeError:
                # Approach 3: Try to extract individual question-answer pairs using regex
                try:
                    # Look for patterns like {"question": "...", "answer": "..."}
                    qa_pattern = r'\{\s*"question":\s*"([^"]+)",\s*"answer":\s*"([^"]+)"\s*\}'
                    matches = re.findall(qa_pattern, result)
                    if matches:
                        qa_pairs = [{"question": q, "answer": a} for q, a in matches]
                        print(f"Extracted {len(qa_pairs)} QA pairs using regex pattern for chunk {i+1}")
                    else:
                        # Approach 4: Look for natural language patterns
                        q_pattern = r'Question:?\s*(.*?)\s*\n+\s*Answer:?\s*(.*?)(?=\n+\s*Question:?|\Z)'
                        nl_matches = re.findall(q_pattern, result, re.DOTALL)
                        if nl_matches:
                            qa_pairs = [{"question": q.strip(), "answer": a.strip()} for q, a in nl_matches]
                            print(f"Extracted {len(qa_pairs)} QA pairs from natural language for chunk {i+1}")
                except Exception as e:
                    print(f"Failed to extract QA pairs from chunk {i+1}: {e}")
        
        # Filter out template/placeholder questions
        filtered_pairs = []
        for qa in qa_pairs:
            if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                continue
                
            q = qa["question"]
            a = qa["answer"]
            
            # Skip template questions and answers
            if any([
                # Only catch obvious template patterns
                "[" in q and "]" in q,                                  # Catches [placeholder] patterns
                "[" in a and "]" in a,                                  # Catches [placeholder] in answers
                "specific" in q.lower() and "document" in q.lower(),   # Specific...document template
                "document states that" in a.lower(),                   # Document states pattern
                "document explains that" in a.lower(),                 # Document explains pattern
                
                # Very short questions/answers are probably not useful
                len(q.strip()) < 8,
                len(a.strip()) < 5
            ]):
                print(f"Filtered out template question: {q}")
                continue
                
            filtered_pairs.append(qa)
            
        # Add valid pairs to the collection
        if filtered_pairs:
            all_qa_pairs.extend(filtered_pairs)
        else:
            print(f"Warning: No valid QA pairs extracted from chunk {i+1}")
    
    # Remove duplicate questions
    seen_questions = set()
    unique_qa_pairs = []
    
    for qa in all_qa_pairs:
        if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
            continue
            
        question = qa["question"]
        if question not in seen_questions:
            seen_questions.add(question)
            unique_qa_pairs.append(qa)
    
    # Return results limited to num_questions per document
    num_pairs = min(num_questions, len(unique_qa_pairs))
    print(f"Extracted {len(unique_qa_pairs)} unique QA pairs from document, limiting to {num_pairs}")
    
    # If we have more pairs than needed, select the most diverse set
    if len(unique_qa_pairs) > num_questions and len(unique_qa_pairs) > 1:
        # Simple approach: take pairs from throughout the list for better diversity
        step = len(unique_qa_pairs) / num_questions
        selected_pairs = []
        for i in range(num_questions):
            idx = min(int(i * step), len(unique_qa_pairs) - 1)
            selected_pairs.append(unique_qa_pairs[idx])
        return selected_pairs
    
    return unique_qa_pairs[:num_questions]


def process_document(model, doc_path: str, questions_per_doc: int = 5, 
                 chunk_size: int = 4000, chunk_overlap: int = 500) -> List[Dict[str, str]]:
    """
    Process a single document file to generate question-answer pairs.
    
    Args:
        model: The LLaMA model instance
        doc_path: Path to the document file
        questions_per_doc: Number of questions to generate per document
        chunk_size: Maximum size of document chunks in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of question-answer pairs in the format needed for fine-tuning
    """
    try:
        # Read the document
        with open(doc_path, 'r', encoding='utf-8') as file:
            doc_text = file.read()
        
        if not doc_text or len(doc_text) < 100:
            print(f"Skipping {doc_path} - document too short")
            return []
        
        # Extract title if available
        title_match = re.search(r'^# (.+)$', doc_text, re.MULTILINE)
        doc_title = title_match.group(1) if title_match else os.path.basename(doc_path)
        
        # Generate QA pairs using LLaMA
        qa_pairs = extract_questions_from_document_with_llama(
            model, 
            doc_text, 
            questions_per_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not qa_pairs:
            print(f"Warning: No QA pairs generated for {doc_path}")
            return []
            
        print(f"Successfully generated {len(qa_pairs)} QA pairs for {doc_path}")
        
        # Format QA pairs for fine-tuning
        formatted_pairs = []
        for qa in qa_pairs:
            question = qa["question"]
            answer = qa["answer"]
            
            # Format with question and answer fields only
            formatted_pair = {
                "question": question,
                "answer": answer
            }
            formatted_pairs.append(formatted_pair)
        
        return formatted_pairs
    
    except Exception as e:
        print(f"Error processing document {doc_path}: {e}")
        import traceback
        traceback.print_exc()
        return []


def iterate_documents(model, docs_dir: str, output_file: str, fields: List[str], 
                     questions_per_doc: int = 5, test_split: float = 0.1,
                     limit: int = None, chunk_size: int = 4000, chunk_overlap: int = 500):
    """
    Iterate through document directories and create a dataset for fine-tuning.
    
    Args:
        model: The LLaMA model instance
        docs_dir: Path to the documents directory
        output_file: Path to save the output dataset
        fields: List of fields to process (corresponds to subdirectories)
        questions_per_doc: Number of questions to generate per document
        test_split: Portion of data to use as test set (0-1)
        limit: Maximum number of documents to process per field (None for all)
        chunk_size: Maximum size of document chunks in characters
        chunk_overlap: Number of characters to overlap between chunks
    """
    all_qa_pairs = []
    docs_processed = 0
    docs_failed = 0
    
    for field in fields:
        field_dir = os.path.join(docs_dir, field)
        if not os.path.exists(field_dir):
            print(f"Field directory not found: {field_dir}")
            continue
        
        print(f"Processing field: {field}")
        
        # Get all document files in the field directory
        doc_files = []
        for root, _, files in os.walk(field_dir):
            for file in files:
                if file.endswith('.txt') and not file.endswith('.txt:Zone.Identifier'):
                    doc_files.append(os.path.join(root, file))
        
        # Apply limit if specified
        if limit is not None:
            doc_files = doc_files[:limit]
            print(f"Limiting to first {limit} documents")
            
        if not doc_files:
            print(f"No document files found in {field}")
            continue
            
        print(f"Found {len(doc_files)} document files in {field} to process")
        
        # Process each document
        for i, doc_file in enumerate(doc_files):
            try:
                print(f"Processing document {i+1}/{len(doc_files)}: {doc_file}")
                qa_pairs = process_document(
                    model, 
                    doc_file, 
                    questions_per_doc, 
                    chunk_size, 
                    chunk_overlap
                )
                
                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    docs_processed += 1
                    
                    # Save intermediate results to avoid losing progress
                    if len(all_qa_pairs) >= 10 and len(all_qa_pairs) % 10 == 0:
                        interim_file = output_file.replace('.json', f'_interim_{len(all_qa_pairs)}.json')
                        os.makedirs(os.path.dirname(interim_file), exist_ok=True)
                        with open(interim_file, 'w', encoding='utf-8') as f:
                            json.dump(all_qa_pairs, f, indent=2)
                        print(f"Saved {len(all_qa_pairs)} examples to interim file {interim_file}")
                else:
                    docs_failed += 1
            except Exception as e:
                print(f"Error processing document {doc_file}: {e}")
                import traceback
                traceback.print_exc()
                docs_failed += 1
                continue
    
    if not all_qa_pairs:
        print("No question-answer pairs were generated. Please check your documents and model.")
        return
        
    print(f"Generated {len(all_qa_pairs)} question-answer pairs from {docs_processed} documents")
    print(f"Failed to process {docs_failed} documents")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Split into train and test if needed
    if test_split > 0 and len(all_qa_pairs) > 5:  # Need at least a few examples
        random.shuffle(all_qa_pairs)
        test_size = int(len(all_qa_pairs) * test_split)
        train_data = all_qa_pairs[test_size:]
        test_data = all_qa_pairs[:test_size]
        
        # Save train data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} training examples to {output_file}")
        
        # Save test data
        test_output = output_file.replace('.json', '_test.json')
        with open(test_output, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        print(f"Saved {len(test_data)} test examples to {test_output}")
    else:
        # Save all data to single file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, indent=2)
        print(f"Saved {len(all_qa_pairs)} examples to {output_file}")


def main():
    """Parse arguments and prepare the data for fine-tuning."""
    parser = argparse.ArgumentParser(description="Generate question-answer pairs from documents for fine-tuning")
    parser.add_argument("--docs_dir", type=str, default="data/raw/documents",
                        help="Directory containing the document collections")
    parser.add_argument("--output_file", type=str, default="data/finetune/instruction_data.json",
                        help="Output file for the fine-tuning dataset")
    parser.add_argument("--model_class", type=str, default="BaselineLLaMA",
                        help="Model class to use for generation (BaselineLLaMA recommended)")
    parser.add_argument("--fields", type=str, nargs='+', 
                        default=["Bio-Medical", "Science"],
                        help="Fields (subdirectories) to process")
    parser.add_argument("--questions_per_doc", type=int, default=5,
                        help="TOTAL number of questions to generate per document (not per chunk)")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Portion of data to use as test set (0-1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of documents to process per field")
    parser.add_argument("--chunk_size", type=int, default=4000,
                        help="Maximum size of document chunks in characters")
    parser.add_argument("--chunk_overlap", type=int, default=500,
                        help="Number of characters to overlap between chunks")
    
    args = parser.parse_args()
    
    # Use paths from centralized path utilities
    docs_dir = DATA_DIR / "raw" / "documents"
    if args.docs_dir != "data/raw/documents":
        # If user specified a custom path, resolve it relative to DATA_DIR
        docs_dir = DATA_DIR / args.docs_dir.lstrip("data/")
    
    # Output file path
    output_file = DATA_DIR / "finetune" / "instruction_data.json"
    if args.output_file != "data/finetune/instruction_data.json":
        # If user specified a custom path, resolve it relative to DATA_DIR
        output_file = DATA_DIR / args.output_file.lstrip("data/")
    
    # Ensure output directory exists
    os.makedirs(output_file.parent, exist_ok=True)
    
    # Create model config container object
    class ModelArgs:
        pass
    
    model_args = ModelArgs()
    model_args.model_config = load_model_config()
    model_args.model_class = args.model_class
    
    # Create the model
    print(f"Creating {args.model_class} model...")
    model = create_model(model_args)
    
    if not hasattr(model, 'generate_ft_data'):
        raise AttributeError(f"The model class {args.model_class} does not have a generate_ft_data method. Use BaselineLLaMA instead.")
    
    iterate_documents(
        model,
        str(docs_dir), 
        str(output_file), 
        args.fields,
        args.questions_per_doc,
        args.test_split,
        args.limit,
        args.chunk_size,
        args.chunk_overlap
    )


if __name__ == "__main__":
    main()