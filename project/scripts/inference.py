import argparse
from pathlib import Path
import yaml
from models import get_model 
from models.pipeline import HallucinationEvalPipeline
from scripts.evaluate import HallucinationEvaluator
from models.fact_bot import BaseLLaMA

class ConfigLoader:
    @staticmethod
    def load(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def get_model_config(config, model_type):
        return config['models'][model_type]

class CLI:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="LLaMA Hallucination Evaluation")
        parser.add_argument('--model', type=str, required=True, 
                          choices=['baseline', 'rag', 'finetuned'],
                          help='Model type to evaluate')
        parser.add_argument('--dataset', type=str, required=True,
                          choices=['HaluEval2', 'FACTBENCH'],
                          help='Dataset to use for evaluation')
        parser.add_argument('--config', type=str, default='configs/main.yaml',
                          help='Path to configuration file')
        return parser.parse_args()

def main():
    args = CLI.parse_args()
    config = ConfigLoader.load(args.config)
    
    # Model initialization
    model_config = ConfigLoader.get_model_config(config, args.model)
    judge_model_config = ConfigLoader.get_model_config(config, 'judge')

    judge_model = BaseLLaMA(judge_model_config)
    model = BaseLLaMA(model_config)
    
    # Pipeline setup
    data_path = Path(config['data_path']) / f"{args.dataset}/Bio-Medical.json"
    save_path = Path(config['results_path']) / args.model / f"{args.dataset}_results.json"
    
    pipeline = HallucinationEvalPipeline(
        model=model,
        judge_model=judge_model,
        data_path=str(data_path),
        save_path=str(save_path)
    )
    
    pipeline.run_evaluation()
    
    results = pipeline.save_data
    ratings = [r['evaluation']['rating'] for r in results]
    print("\nEvaluation Summary:")
    print(f"Average rating: {sum(ratings)/len(ratings):.2f}")
    print(f"Number of complete hallucinations (rating=1): {ratings.count(1)}")
    print(f"Number of factual responses (rating=5): {ratings.count(5)}")

if __name__ == "__main__":
    main()