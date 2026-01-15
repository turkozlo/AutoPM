import os

from sentence_transformers import SentenceTransformer


def download_model(model_name, save_directory):
    """
    Downloads a SentenceTransformer model and saves it to a local directory.
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    model_path = os.path.join(save_directory, model_name)

    if os.path.exists(model_path):
        print(f"Model '{model_name}' already exists in '{save_directory}'.")
        return

    print(f"Downloading model '{model_name}' to '{model_path}'...")
    model = SentenceTransformer(model_name)
    model.save(model_path)
    print(f"Model '{model_name}' successfully saved to '{model_path}'.")


if __name__ == "__main__":
    # Default model used in rag.py
    MODEL_NAME = "all-MiniLM-L6-v2"
    SAVE_DIR = "models"

    download_model(MODEL_NAME, SAVE_DIR)
