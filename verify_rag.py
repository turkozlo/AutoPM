import os

from pm_agent.rag import RAGManager


def test_rag():
    print("--- Starting RAG Verification ---")

    # 1. Initialize RAG
    rag = RAGManager()

    # 2. Load test Excel file
    excel_file = "processes.xlsx"
    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found. Run generate_test_data.py first.")
        return

    success = rag.load_excel(excel_file)
    if not success:
        print("Failed to load Excel file.")
        return

    # 3. Test Query
    query = "What is process P101?"
    print(f"\nQuerying: '{query}'")
    context = rag.get_context_string(query)
    print(context)

    if "Loan Application" in context:
        print(
            "\n✅ Verification SUCCESS: RAG correctly retrieved 'Loan Application' for P101."
        )
    else:
        print(
            "\n❌ Verification FAILED: RAG did not retrieve the expected information."
        )


if __name__ == "__main__":
    test_rag()
