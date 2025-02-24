from typing import List
import json
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from torch.utils.data import Dataset, DataLoader

class QuestionDataset(Dataset):
    def __init__(self, json_file):
        # Load the JSON file which is expected to be a list of dictionaries.
        with open(json_file, 'r') as f:
            self.questions = json.load(f)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Retrieve a single sample (a dictionary) from the dataset.
        sample = self.questions[idx]
        
        # Extract the desired fields from the sample
        question_id = sample['id']
        user_query = sample['user_query']
        answers = sample.get('local_llm_answers', "")  
        facts = sample.get('facts', "") 
        judge = sample.get('judge', "") 
        
        # Return all the required fields
        return question_id, user_query, answers, facts, judge

    
    def update_sample(self, question_id, key, value):
        """Update a specific sample in the dataset."""
        for sample in self.questions:
            if sample['id'] == question_id:
                sample[key] = value
                break


class HallucinationEvalPipeline:
    """Evaluation pipeline for hallucination analysis"""
    def __init__(self, test_model, eval_model, opt):
        self.test_model = test_model
        self.eval_model = eval_model
        self.config = opt.pipeline_config
        self.dataset = QuestionDataset(f'data/{opt.dataset}/{opt.field}.json')
        self.save_path = f"results/{opt.field}.json"

    def load_data(self):
        """Custom data loader"""
        return DataLoader(
            self.dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config['num_workers']
        )

    def generate_answers(self):
        """Generate answers for the questions in the dataset using the test model."""
        data_loader = self.load_data()
        for batch in tqdm(data_loader, desc="Generating Answers"):
            question_ids, user_queries, *_ = batch

            for question_id, user_query in zip(question_ids, user_queries):
                answer = self.test_model.generate(user_query)
                answer = self.extract_response(answer)
                self.dataset.update_sample(question_id, "local_llm_answers", answer)

            del batch  # Free batch memory
            #saving the results
            self.save_results()

        del data_loader


    def generate_answers_batches(self):
        """Generate answers in batches for the questions in the dataset using the test model."""
        data_loader = self.load_data()
        for batch in tqdm(data_loader, desc="Generating Answers"):
            question_ids, user_queries, *_ = batch

            answers = self.test_model.generate_batches(user_queries)
            for question_id, answer in zip(question_ids, answers):
                answer = self.extract_response(answer)
                self.dataset.update_sample(question_id, "local_llm_answers", answer)

            del batch
            #saving the results
            self.save_results()
        
        del data_loader
            


    def generate_facts(self):
        """Generate facts for the questions in the dataset using the eval model."""
        data_loader = self.load_data()
        with open(self.config['fact_prompt_path'], "r") as f:
            template = f.read()

        prompt_template = PromptTemplate(
            input_variables=["query", "answer"],
            template=template
        )

        for batch in tqdm(data_loader, desc="Generating Facts"):
            question_ids, user_queries, answers, *_ = batch
            for question_id, user_query, answer in zip(question_ids, user_queries, answers):
                prompt = prompt_template.format(query=user_query, answer=answer)
                facts = self.eval_model.generate(prompt)
                #print(facts)
                #facts_lst = self.get_facts_lst(facts)
                self.dataset.update_sample(question_id, "facts", facts)

            del batch  # Free batch memory
            #saving the results
            self.save_results()
        
        del data_loader


    def evaluate_facts(self):
        """Evaluate the generated facts using the eval model."""
        data_loader = self.load_data()
        with open(self.config['judge_prompt_path'], "r") as f:
            template = f.read()

        prompt_template = PromptTemplate(
            input_variables=["facts"],
            template=template
        )

        for batch in tqdm(data_loader, desc="Evaluating Facts"):
            question_ids, _, _, facts, _ = batch
            for question_id, facts_lst in zip(question_ids, facts):
                prompt = prompt_template.format(facts=facts_lst)
                judge = self.eval_model.generate(prompt)
                self.dataset.update_sample(question_id, "judge", judge)

            del batch  # Free batch memory
            #saving the results
            self.save_results()

        del data_loader

    def get_facts_lst(self, response: str) -> List[str]:
        """Extract facts list from the LLM response."""
        if not response or "NO FACTS" in response:
            return []
        
        try:
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            if not lines:
                print(f"Empty facts: {response}")
                return []
                
            if len(lines) == 1 and not lines[0].startswith("1."):
                return [lines[0]]
                
            return [fact[2:].strip() for fact in lines if fact[2:].strip()]
            
        except Exception as e:
            print(f"Error parsing facts: {str(e)}")
            print(f"Response: {response}")
            return []


    def extract_response(self, full_text: str) -> str:
        """Extracts and returns only the portion of the response following the "### Response:" marker."""
        marker = "### Response:"
        if marker in full_text:
            # Split the text at the marker and return the part after it, stripped of leading/trailing whitespace
            return full_text.split(marker, 1)[1].strip()
        else:
            # Marker not found, return the whole text stripped
            return full_text.strip()
        

    def save_results(self):
        """Save the dataset with the generated data."""
        with open(self.save_path, 'w') as f:
            json.dump(self.dataset.questions, f, indent=2)
