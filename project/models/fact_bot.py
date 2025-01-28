# coding: utf-8
import os
from tqdm import tqdm
from response import Chatbot, Parser, check_exist
from models import BaseLLM

class Factbot(BaseLLM):
    """
    Chatbot for factual statements generation.
    """

    def __init__(self, data_path, save_path, model, file, assist_model):
        super().__init__(data_path, save_path, model, file)
        self.file = file  # file name
        self.assist_model = assist_model  # facts generation model

    def get_facts_lst(self, ans):
        """
        Get facts list from the assist model's response.
        """
        if "NO FACTS" in ans:
            return []
        try:
            lines = [line.strip() for line in ans.split("\n") if line.strip()]
            if len(lines) == 0:
                print("Empty facts: " + ans)
                return []
            elif len(lines) == 1 and not lines[0].startswith("1."):
                return [lines[0]]
            else:
                return [fact[2:].strip() for fact in lines if fact[2:].strip()]
        except Exception as e:
            print("Error: " + str(e))
            print("Corresponding facts: " + ans)
            return []

    def generate_facts(self, data, prompt, **kwargs):
        """
        Generate facts by the assist model.
        """
        if len(data) == 0:
            return
        for i in tqdm(range(len(data)), ncols=100):
            if len(self.save_data) % self.frequency == 0:
                self.save()
            query = prompt.format(
                query=data[i]["user_query"], answer=data[i][self.model + "_response"]
            )
            ans = self.openai_complete(query, self.assist_model, **kwargs)
            data[i][self.model + "_fact_raw"] = ans
            ans = self.post_process(ans)
            facts = self.get_facts_lst(ans)
            data[i][self.model + "_fact"] = facts
            self.save_data.append(data[i])

