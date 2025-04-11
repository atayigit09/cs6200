#!/bin/bash
# Script to install required packages for 4-bit quantization

# Check if running in virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
  echo "Please activate your virtual environment first with:"
  echo "source venv/bin/activate"
  exit 1
fi

# Install required packages
pip install torch>=2.0.0
pip install bitsandbytes>=0.41.1
pip install accelerate>=0.22.0
pip install transformers>=4.31.0
pip install peft>=0.4.0
pip install dotenv

echo "Installation complete. Now run your script with:"
echo "python -m project.scripts.prepare_finetune_data --limit 2" 