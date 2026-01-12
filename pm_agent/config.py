import yaml
import os
from pathlib import Path

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {CONFIG_PATH}\n"
            f"Please copy config.example.yaml to config.yaml and fill in your settings."
        )
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()

# Provider setting: "mistral" or "local"
PROVIDER = config.get("provider", "mistral")

# Extract Mistral Config (cloud)
mistral_config = config.get("mistral", {})
MISTRAL_API_KEY = mistral_config.get("api_key")
MISTRAL_MODEL = mistral_config.get("model", "mistral-small-latest")

# Extract Local Model Config (OpenAI-compatible API)
local_config = config.get("local", {})
LOCAL_BASE_URL = local_config.get("base_url", "http://localhost:11434/v1")
LOCAL_MODEL = local_config.get("model", "llama3.2")
LOCAL_API_KEY = local_config.get("api_key", "ollama")

# Validate config based on provider
if PROVIDER == "mistral":
    if not MISTRAL_API_KEY:
        raise ValueError("Mistral API Key not found in configuration (mistral.api_key).")
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
elif PROVIDER == "local":
    if not LOCAL_BASE_URL:
        raise ValueError("Local model base_url not found in configuration (local.base_url).")
else:
    raise ValueError(f"Unknown provider: {PROVIDER}. Use 'mistral' or 'local'.")
