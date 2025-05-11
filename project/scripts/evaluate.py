import argparse
import yaml
from pathlib import Path

from models import create_model
from models.evaluator import EvalLLM  
from evaluation.pipeline import HallucinationEvalPipeline


def load_model_config():
    """Loads the configuration file based on the model_name."""
    config_path = Path(f"configs/base_model.yaml").resolve()

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

    parser.add_argument("--model_class", type=str, required=True, 
                        choices=['BaselineLLaMA','LoraLLaMA', 'RagLLaMA'],
                        help="Class of the model")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['HaluEval2'],
                        help='Dataset to use for evaluation')
    parser.add_argument('--field', type=str, default='Open-Domain',
                        choices=['Bio-Medical','Education','Finance','Open-Domain','Science', 'test'],
                        help='Dataset field to use for evaluation')

    args = parser.parse_args()
    
    # Load config dynamically
    args.model_config = load_model_config()
    args.eval_config = load_eval_config()
    args.pipeline_config = load_pipeline_config()

    return args


if __name__ == "__main__":
    opt = parse_args()

    print("loading test LLm")
    test_llm = create_model(opt)

    print("loading Eval LLm")
    eval_llm = EvalLLM(opt.eval_config)

    print("loading pipeline")
    pipeline = HallucinationEvalPipeline(test_llm, eval_llm, opt)

    #pipeline.generate_answers_batches() ### THIS should be used when using clusters!!
    pipeline.generate_answers()
    pipeline.generate_facts()
    pipeline.evaluate_facts()
    


    # ##TESTING
    # query = "Can you clarify whether the claim \"10% of sudden infant death syndrome (SIDS) deaths happen in newborns aged less than 6 months.\" is accurate or not? Build factual arguments about the claim."
 

    # docs = test_llm.search(
    #     query=query,
    # )

    # print(docs)
    # results = test_llm.format_context(
    #     documents=docs,
    # )

    # print(results)


