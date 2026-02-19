import argparse
import pandas as pd
import sys
import os
import warnings

# Suppress warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Fix python path to allow running as script from root or subdirectory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pm_agent.llm import LLMClient
from pm_agent.agents.formatter import DataFormatterAgent

def main():
    parser = argparse.ArgumentParser(description="AutoPM Agent (Minimal)")
    parser.add_argument("--file", type=str, help="Path to event log file")
    args = parser.parse_args()

    # 1. Init
    print("🤖 Initializing AutoPM Agent (Minimal Mode)...")
    try:
        llm_client = LLMClient()
    except Exception as e:
        print(f"❌ Error initializing LLM Client: {e}")
        return

    # 2. Load Data
    file_path = args.file
    if not file_path:
        print("Please provide a file using --file argument or input below.")
        file_path = input("Enter path to dataset file: ").strip().strip('"')
        
    print(f"📂 Loading data from {file_path}...")
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
             # Try csv default
             df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print(f"✅ Data loaded.")

    # 3. Format Data
    print("🧹 Formatting data types using DataFormatter...")
    try:
        formatter = DataFormatterAgent(df, llm_client)
        df = formatter.run()
        print("✅ Data formatting complete.")
    except Exception as e:
        print(f"⚠️ Formatting failed: {e}. Proceeding with raw data.")

    # 4. Generate Basic Report
    rows = len(df)
    cols = len(df.columns)
    col_names = ", ".join(list(df.columns))
    
    report_content = (
        "# Basic Data Report\n\n"
        f"- **Rows**: {rows}\n"
        f"- **Columns**: {cols}\n"
        f"- **Column Names**: {col_names}\n"
    )
    
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "report.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"📄 Report saved to {report_path}")
    print("\n--- REPORT PREVIEW ---")
    print(report_content)
    print("----------------------")

    # 5. Simple Chat Loop
    print("\n💬 Chat with your data structure! (Type 'exit' to quit)")
    context_str = f"Rows: {rows}, Columns: {cols}. Column Names: {col_names}."
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            if not user_input:
                continue
                
            print("🤖 Thinking...")
            response = llm_client.simple_chat(user_input, context_str)
            print(f"🤖 {response}")
                
        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    main()
