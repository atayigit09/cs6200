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
📌 **Note**: You need **Hugging Face access** to the `meta/llama` repository. If you don’t have access, please request it soon.

