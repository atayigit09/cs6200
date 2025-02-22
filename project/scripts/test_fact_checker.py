import yaml
import os
from pathlib import Path
import json
import asyncio
from models import EvalLLM

def interpolate_env_vars(config):
    """Replace ${VAR} with environment variable values"""
    if isinstance(config, dict):
        return {k: interpolate_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [interpolate_env_vars(i) for i in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var)
    return config

async def main():
    # Get project root directory and config paths
    project_root = Path(__file__).parent.parent
    main_config_path = project_root / "configs" / "main.yaml"
    fact_checker_path = project_root / "configs" / "fact_checker.yaml"

    # Ensure environment variables are set
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not set"
    assert "ANTHROPIC_API_KEY" in os.environ, "ANTHROPIC_API_KEY not set"

    # Load configs and interpolate environment variables
    with open(main_config_path) as f:
        main_config = yaml.safe_load(f)
        
    with open(fact_checker_path) as f:
        fact_config = yaml.safe_load(f)
        fact_config = interpolate_env_vars(fact_config)

    # Test data
    test_data = [
        {
            "user_query": "Who was Albert Einstein?",
            "model_response": "Albert Einstein was born in 1879 in Ulm, Germany. He developed the theory of relativity and won the Nobel Prize in Physics in 1921. He moved to the United States in 1933 and worked at Princeton University until his death in 1955."
        }
    ]

    # Create test data directory relative to project root
    test_data_dir = project_root / "data" / "test"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(test_data_dir / "test_responses.json", "w") as f:
        json.dump(test_data, f, indent=2)

    # Test both fact checkers
    for checker_name, checker_config in fact_config['fact_checkers'].items():
        print(f"\nTesting {checker_name} fact checker...")
        
        # Load prompts using absolute paths
        extraction_path = project_root / checker_config['prompts']['fact_extraction']
        validation_path = project_root / checker_config['prompts']['fact_validation']
        
        with open(extraction_path) as f:
            extraction_prompt = f.read()
        with open(validation_path) as f:
            validation_prompt = f.read()
        
        # Initialize EvalLLM
        eval_model = EvalLLM(checker_config)
        
        # Test fact extraction
        print("\nExtracting facts...")
        results = await eval_model.generate_facts(test_data, extraction_prompt)
        
        print(f"\nResults from {checker_name}:")
        for item in results:
            print("\nExtracted facts:")
            for fact in item['extracted_facts']:
                print(f"- {fact}")
            
            # Validate facts
            print("\nValidating facts...")
            facts_text = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(item['extracted_facts']))
            validation_text = validation_prompt.format(facts=facts_text)
            
            validation_response = await eval_model.generate_completion(validation_text)
            print("\nValidation results:")
            print(validation_response)

    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(main()) 