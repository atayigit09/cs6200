import argparse
import yaml
from pathlib import Path
from models.evaluator import EvalLLM  
import yaml
from evaluation.pipeline import HallucinationEvalPipeline
def load_model_config(model_name):
    """Loads the configuration file based on the model_name."""
    config_path = Path(f"configs/{model_name}.yaml").resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found!")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

def load_eval_config():
    with open("configs/evaluator.yaml", "r") as file:
        eval_config = yaml.safe_load(file)
    return eval_config

def load_pipeline_config():
    with open("configs/pipeline.yaml", "r") as file:
        pipeline_config = yaml.safe_load(file)
    return pipeline_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hallucination Evaluation")

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['HaluEval2'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--field', type=str, default='Open-Domain',
                        choices=['Bio-Medical','Education','Finance','Open-Domain','Science', 'test'],
                        help='Dataset field to use for evaluation')

    args = parser.parse_args()
    
    # Load config dynamically
    args.eval_config = load_eval_config()
    args.pipeline_config = load_pipeline_config()

    return args


if __name__ == "__main__":
    opt = parse_args()

    print(opt)
    # print("loading test LLm")
    # model = create_model(opt)

    test_llm = EvalLLM(opt.eval_config) #testing purpose so not using local llm

    print("loading Eval LLm")
    eval_llm = EvalLLM(opt.eval_config)


    print("loading pipeline")

    pipeline = HallucinationEvalPipeline(test_llm, eval_llm, opt)
    pipeline.process()

    
