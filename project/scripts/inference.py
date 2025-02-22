import argparse
from pathlib import Path
import yaml
from models import get_model, BaselineLLaMA, HallucinationEvalPipeline

class CLI:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description="LLaMA Hallucination Evaluation")
        parser.add_argument('--model', type=str, required=True, 
                          choices=['baseline', 'rag', 'finetuned'],
                          help='Model type to evaluate')
        parser.add_argument('--dataset', type=str, required=True,
                          choices=['HEVAL2.0', 'FACTBENCH'],
                          help='Dataset to use for evaluation')
        parser.add_argument('--config', type=str, default='configs/main.yaml',
                          help='Path to configuration file')
        return parser.parse_args()

def main():
    args = CLI.parse_args()
    config = ConfigLoader.load(args.config)
    
    # Model initialization
    model_config = ConfigLoader.get_model_config(config, args.model)
    if args.model == 'baseline':
        model = BaselineLLaMA(model_config)
    
    # Pipeline setup
    data_path = Path(config['data_path']) / f"{args.dataset}.json"
    save_path = Path(config['results_path']) / args.model / f"{args.dataset}_results.json"
    
    pipeline = HallucinationEvalPipeline(
        model=model,
        data_path=str(data_path),
        save_path=str(save_path)
    )
    
    pipeline.run_evaluation()

if __name__ == "__main__":
    main()