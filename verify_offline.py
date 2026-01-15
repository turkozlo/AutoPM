from pm_agent.rag import RAGManager


def test_local_loading():
    print("Testing local model loading...")
    # Initialize RAGManager without explicit path - it should find it in 'models/all-MiniLM-L6-v2'
    try:
        rag = RAGManager()
        print(f"Successfully loaded model from: {rag.model_path}")

        # Check if the path is indeed local
        if "models" in rag.model_path:
            print("SUCCESS: Model loaded from local 'models' directory.")
        else:
            print("FAILURE: Model loaded from Hugging Face or unexpected path.")

    except Exception as e:
        print(f"ERROR during model loading: {e}")


if __name__ == "__main__":
    test_local_loading()
