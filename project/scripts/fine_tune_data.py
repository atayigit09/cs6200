import os
import json
import argparse
from ollama import chat
from ollama import ChatResponse
import re

def chunk_text(text, chunk_size=1000):
    """Splits the text into smaller chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def format_prompt(document_content):
    prompt_template = f"""
            You are provided with the following document:

            \"\"\"
            {document_content}
            \"\"\"

            Your task is to extract straightforward, fact-based but detailed questions and their corresponding answers solely from the content of the document. Follow these rules:

            1. **Source Strictness:** Only use the information explicitly provided in the document. Do not use any internal or external knowledge.
            2. **Extraction:** Generate questions that cover key details of the document. Each question must have a corresponding answer that is clearly supported by the document's content.
            3. **Clarity:** Ensure that the questions are clear and unambiguous, allowing for straightforward answers.
            4. **Question Styles**: Use a variety of question styles, including but not limited to:
                - True false questions
                - What is/are questions
                - How is/are questions
            5. **Quantity:** Produce at most 20 questions. If the document contains fewer key details, generate fewer questions accordingly.
            6. **Format:** Output your results in valid JSON format, structured as an array of objects, where each object has two keys: "question" and "answer".

            Example output format:
            [
            {{
                "question": "Question text here?",
                "answer": "Answer text here."
            }},
            ...
            ]

            Do not include any extra text or commentary. Provide only the JSON output.
            """
    return prompt_template


def main():
    parser = argparse.ArgumentParser(description="Document Scraper")
    parser.add_argument("--field", 
                        choices=["Bio-Medical", "Education", "Finance", "Open-Domain", "Science", "test"],
                        required=True,
                        help="Select the topic file to process")
    parser.add_argument("--results-dir", 
                        default="project/data/fineTune",
                        help="Directory containing the results JSON files")
    

    args = parser.parse_args()


    data_folder = f"data/docs/{args.field}"
    save_folder = f"data/fineTune/{args.field}"
    

    #saving empty json file
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #create a json file for the field
    json_file = os.path.join(save_folder, f"{args.field}.json")
    if not os.path.exists(json_file):
        with open(json_file, "w") as f:
            json.dump([], f)

    qa_pattern = re.compile(
        r'\{\s*"question"\s*:\s*"([^"]+?)"\s*,\s*"answer"\s*:\s*"([^"]+?)"\s*\}',
        re.DOTALL
    )

    #read the txt file for every file in the folder
    for file_name in os.listdir(data_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, "r") as f:
                data = f.read()
                print(f"Processing {file_name}...")

                prompt = format_prompt(data)

                response: ChatResponse = chat(model='llama3.2:1b', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
                ])
                qas = response['message']['content']

                qas = qas.strip()

                # Extract all QA objects
                qa_matches = qa_pattern.findall(qas)

                parsed_objects = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_matches]

                # Append the parsed JSON to the output file
                with open(json_file, "r+") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
                    existing_data.extend(parsed_objects)
                    f.seek(0)
                    json.dump(existing_data, f, indent=4)

            
    
if __name__ == "__main__":
    main()
