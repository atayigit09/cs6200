from models import BaseLLM
import json

class HallucinationEvaluator:
    def __init__(self, model: BaseLLM, judge_model: BaseLLM):
        self.model = model
        self.judge_model = judge_model
        self.prompt_files = {"Evaluate": "../prompts/evaluation/determine_truthfulness.txt",
                             "Fact": "../prompts/evaluation/generate_facts.txt"}
        self.prompts = self.load_prompts()

    def load_prompts(self) -> dict:
        """Load prompts from files"""
        prompts = {}
        for mode, file in self.prompt_files.items():
            with open(file, 'r') as f:
                prompts[mode] = f.read()
        return prompts

    def __get_fact(self, question: str) -> str:
        answer = self.model.generate(question)
        prompt = self.prompts['Fact']
        return self.model.generate(prompt.format(question, answer))

    def __evaluate_fact(self, fact: str) -> str:
        prompt = self.prompts['Evaluate']
        return self.judge_model.generate(prompt.format(fact))

    def evaluate_response(self, question: str, response: str) -> dict:
        """Evaluate the response using the judge model"""
        prompt = self.format_judge_prompt(question, response)
        judge_response = self.judge_model.generate_text(prompt)

        try:
            evaluation = json.loads(judge_response)
            return evaluation
        except json.JSONDecodeError:
            print(f"Error parsing JSON response: {judge_response}")
            return {
                "rating": 0,
                "explanation": "Could not parse JSON response",
                "hallucinated_claims": []
            }