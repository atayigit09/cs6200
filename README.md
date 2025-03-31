# CS6200

## Project Layout

### 📂 Model Folder
Each model is a wrapper of the base LLM. Current implementations include:
- `base-llama`: The foundational LLaMA model With BaseLine class and RAG class.
- `evaluator`: A model used for evaluation.
- `embeddings`: Embedding folder with various classes for generating embeddings.

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

#### Base Model Configuration Format
The configuration file (`configs/base_model.yaml`) defines all parameters for the model, fine-tuning, and RAG functionality. Below is an explanation of the configuration sections:

##### Model Configuration
```yaml
model:
  model_id: meta-llama/Llama-3.2-1B-Instruct  # HuggingFace model ID
  use_fast: true  # Whether to use fast tokenizer
```

##### Quantization Options
```yaml
quantization:
  load_in_8bit: false  # Enable 8-bit quantization
  load_in_4bit: false  # Enable 4-bit quantization
```

##### Generation Parameters
```yaml
generation:
  max_length: 1000  # Maximum token length for generation
  temperature: 0.7  # Controls randomness (higher = more random)
  top_p: 0.9  # Nucleus sampling parameter
  format_prompt: true  # Whether to format the prompt for the model
```

##### Fine-tuning Parameters
```yaml
finetuning:
  use_qlora: false  # Use QLoRA for quantized fine-tuning
  use_lora: true  # Use LoRA for parameter-efficient fine-tuning
  lora_r: 8  # LoRA rank
  lora_alpha: 32  # LoRA alpha parameter
  target_modules: ["q_proj", "v_proj"]  # Modules to apply LoRA to
  lora_dropout: 0.05  # Dropout rate for LoRA
  bias: none  # Bias type for LoRA
```

##### RAG Configuration
```yaml
rag:
  debug: True  # Enable debug mode
  field: "Science"  # Topic for RAG document storage
  docs_path: "data/docs"  # Path to document directory
  chunk_size: 1024  # Size of document chunks
  chunk_overlap: 100  # Overlap between chunks
  batch_size: 256  # Batch size for processing documents
  
  # Retrieval parameters
  top_k: 5  # Number of documents to retrieve
  
  # Context formatting options
  context_format: "compact"  # Options: "simple" or "compact"
  
  # Prompt template options
  prompt_template: "default"  # Options: "default", "concise", or "custom"
  custom_prompt_template: ""  # Custom template if prompt_template is "custom"
  
  # Vector database configuration
  vector_db:
    type: "faiss"  # Vector storage options: "faiss" or "chroma"
    dimension: 768  # Embedding dimension
    db_path: "data/vector_db"  # Path to save/load the database
    collection_name: "documents"  # For ChromaDB
    persist_directory: "data/chroma_db"  # For ChromaDB
  
  # Embedding model configuration
  embedding:
    type: SentenceTransformerEmbeddings  # Options: SentenceTransformerEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings
    model_name: "ibm-granite/granite-embedding-278m-multilingual"  # HuggingFace embedding model name
    embedding_dimension: 768  # Embedding dimension
    device: "mps"  # Device to run embeddings on (cpu, cuda, mps)
    api_key: ""  # API key for OpenAI embeddings (uses OPENAI_API_KEY env var if empty)
```

### Key Configuration Options

1. **Vector Storage Options**:
   - `faiss`: Facebook AI Similarity Search, an efficient similarity search library. Best for speed and memory efficiency.
   - `chroma`: ChromaDB, a more feature-rich vector database with additional metadata capabilities.

2. **Embedding Model Options**:
   - `SentenceTransformerEmbeddings`: Open-source embedding models from the sentence-transformers library.
   - `OpenAIEmbeddings`: OpenAI's embedding models (requires API key).
   - `HuggingFaceEmbeddings`: Various embedding models from Hugging Face.

3. **Context Formatting**:
   - `simple`: Basic formatting that includes document chunks with minimal formatting.
   - `compact`: More efficient formatting that optimizes token usage.

4. **Prompt Templates**:
   - `default`: Standard RAG prompt template with detailed instructions.
   - `concise`: Minimalist prompt template for shorter contexts.
   - `custom`: User-defined prompt template specified in `custom_prompt_template`.

5. **Quantization Options**:
   - Using 8-bit or 4-bit quantization can reduce memory usage at the cost of some precision.

### 📂 Rag Folder
Contains the code for Scrappers and Document storage models.
- **Scrappers**: Code for scraping documents from various sources.
- **Document Storage**: Code for storing and managing documents in a vector database.

### 📂 Data Folder
Contains datasets and documents for training and evaluation:
- **Datasets**: `data/HaluEval2/`
- **Documents**: `data/docs/`   (Scraped documents)
- **Keywords**: `data/keyWords/` (used for scraping documents)
- **Embeddings**: `data/vector_db/` (Vector database storage)
---

## 🔧 Running Commands

### ✅ Test Run Without Local LLM
```bash
python -m scripts.test --dataset HaluEval2 --field test
```

### 🚀 Local LLM Benchmark
Here model_class can be `BaselineLLaMA` or `RagLLaMA` depending on the model you want to evaluate.
```bash
python -m scripts.evaluate  --model_class BaselineLLaMA --dataset HaluEval2 --field test
```
The evaluation pipeline will:
1. Generate answers using the RAG model
2. Extract facts from the generated answers 
3. Evaluate the factual accuracy of the responses

Results are saved in the `results/` directory.

📌 **Note**: You need **Hugging Face access** to the `meta/llama` repository. If you don't have access, please request it soon.

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




