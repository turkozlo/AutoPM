import json
import os
from pathlib import Path

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# Extract Mistral Config
MISTRAL_API_KEY = config.get("api_key_LANGCHAIN_mistral")
MISTRAL_MODEL = config.get("model_LANGCHAIN_mistral", "mistral-small-latest")

if not MISTRAL_API_KEY:
    raise ValueError("Mistral API Key not found in configuration.")

os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
