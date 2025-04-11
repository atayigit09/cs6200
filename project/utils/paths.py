"""Path utilities for the CS6200 project."""
from pathlib import Path
import os
import sys

# Get the project root directory - this should point to the "project" directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SCIENCE_DATA_DIR = RAW_DATA_DIR / "documents" / "Science"
BIO_DATA_DIR = RAW_DATA_DIR / "documents" / "Bio-Medical"
BIO_CHECKPOINTS_DIR = DATA_DIR / "bio_checkpoints"
FINETUNE_DATA_DIR = DATA_DIR / "finetune"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"

# Configuration directory
CONFIGS_DIR = PROJECT_ROOT / "configs"

# RAG directory
RAG_DIR = PROJECT_ROOT / "rag"

# Add project root to Python path if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def ensure_dir(directory):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory

# Ensure all directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, SCIENCE_DATA_DIR, BIO_DATA_DIR, BIO_CHECKPOINTS_DIR, FINETUNE_DATA_DIR,
                 MODELS_DIR, CONFIGS_DIR, RAG_DIR]:
    ensure_dir(directory) 