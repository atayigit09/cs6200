# CS6200

## Project Layout

### 📂 Model Folder
Each model is a wrapper of the base LLM. Current implementations include:
- `base-llama`: The foundational LLaMA model.
- `evaluator`: A model used for evaluation.

🔹 **Future Work**: We plan to add `llama-fine-tuned` for enhanced performance.

### 📂 Scripts Folder
Contains scripts for running, training, and evaluation. No class definitions should be present here.
- Arguments are provided via the command line interface (CLI).

### 📂 Evaluation Folder
Includes the code for the evaluation pipeline.

### 📂 Configs Folder
Stores configuration files for different components:
- **Pipeline configurations**
- **Evaluation model configurations**
- **Base model configurations**
- **Future Additions**: Fine-tuned model configurations and training configurations.

---

## 🔧 Running Commands

### ✅ Test Run Without Local LLM
```bash
python -m scripts.test --dataset HaluEval2 --field test
```

### 🚀 Local LLM Benchmark
```bash
python -m scripts.evaluate --model_name base_model --model_class BaselineLLaMA --dataset HaluEval2 --field test
```
📌 **Note**: You need **Hugging Face access** to the `meta/llama` repository. If you don't have access, please request it soon.

### 📄 Embedding Documents for RAG
To embed documents for use with the RAG model, run:
```bash
python -m project.scripts.embeddings --data_path data/docs
```
This script will:
1. Load documents from the specified directory
2. Split them into manageable chunks
3. Generate embeddings using the configured embedding model
4. Store the embeddings in a vector database (FAISS or Chroma)

The configuration for embedding is defined in `configs/base_model.yaml` under the `rag` section.

### 🌐 Document Scraping
To scrape documents for specific fields:
```bash
python -m project.scripts.scraper --field [field] --source Wiki --results-dir ./data/docs
```
Where:
- `field`: Choose from Bio-Medical, Education, Finance, Open-Domain, Science, or test
- `source`: Currently supports Wiki (Wikipedia)
- `results-dir`: Directory to save scraped documents

The script uses keywords from `data/keyWords/[field].json` to scrape relevant documents and saves them to the specified directory.

### 📊 Metrics Calculation
To calculate evaluation metrics from the results:
```bash
python -m project.scripts.metrics --field [field] --results-dir ./results --output-excel
```
Where:
- `field`: Same options as above
- `results-dir`: Directory containing the results JSON files
- `--output-excel`: Optional flag to export metrics to Excel files

The script will compute:
- Per-entry metrics (accuracy, false rate, unknown rate, F1-score)
- Aggregate metrics (overall accuracy, macro F1-score)

Results are displayed in the console and optionally exported to Excel files.

### 🔍 Evaluating RAG Model
To evaluate the RAG model:
```bash
python -m project.scripts.evaluate --model_class RagLLaMA --dataset HaluEval2 --field test
```

The evaluation pipeline will:
1. Generate answers using the RAG model
2. Extract facts from the generated answers 
3. Evaluate the factual accuracy of the responses

Results are saved in the `results/` directory.

