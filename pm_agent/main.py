import argparse
import json
import pandas as pd
import sys
import os
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Fix python path to allow running as script from root or subdirectory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pm_agent.llm import LLMClient
from pm_agent.agents.formatter import DataFormatterAgent


# ---------------------------------------------------------------------------
#  Session helpers
# ---------------------------------------------------------------------------

def get_session_dir(file_path: str) -> str:
    """Returns the session directory for a given dataset file."""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join("reports", basename)


def save_session(session_dir: str, df: pd.DataFrame, column_roles: dict, file_path: str):
    """Saves formatted data and column roles to disk."""
    os.makedirs(session_dir, exist_ok=True)

    # Save formatted data
    df.to_csv(os.path.join(session_dir, "formatted_data.csv"), index=False)

    # Save session metadata
    session_meta = {
        "column_roles": column_roles,
        "source_file": os.path.abspath(file_path),
        "saved_at": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "rows": len(df),
        "columns": len(df.columns),
    }
    with open(os.path.join(session_dir, "session.json"), "w", encoding="utf-8") as f:
        json.dump(session_meta, f, ensure_ascii=False, indent=2)

    print(f"💾 Session saved to {session_dir}/")


def load_session(session_dir: str):
    """
    Tries to load a previous session.
    Returns (df, column_roles, meta) or None.
    """
    session_path = os.path.join(session_dir, "session.json")
    data_path = os.path.join(session_dir, "formatted_data.csv")

    if not os.path.exists(session_path) or not os.path.exists(data_path):
        return None

    with open(session_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(data_path)
    return df, meta["column_roles"], meta


# ---------------------------------------------------------------------------
#  Column mapping
# ---------------------------------------------------------------------------

def ask_column(columns: list, role_name: str) -> str:
    """Prompts user to select a column for a given PM role."""
    while True:
        choice = input(f"  {role_name}: ").strip()
        if not choice:
            continue

        # Try as number
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(columns):
                return columns[idx - 1]
            else:
                print(f"    ⚠️ Номер должен быть от 1 до {len(columns)}.")
                continue

        # Try as column name (exact)
        if choice in columns:
            return choice

        # Fuzzy match (case-insensitive)
        matches = [c for c in columns if c.lower() == choice.lower()]
        if matches:
            return matches[0]

        print(f"    ⚠️ Колонка '{choice}' не найдена. Попробуйте ещё раз.")


def map_columns(df: pd.DataFrame) -> dict:
    """Interactive column mapping for Process Mining roles."""
    columns = list(df.columns)

    print("\n📋 Колонки в датасете:")
    print("-" * 40)
    for i, col in enumerate(columns, 1):
        sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        if len(sample) > 30:
            sample = sample[:30] + "..."
        print(f"  {i:3d}. {col:<30s}  (пример: {sample})")
    print("-" * 40)

    print("\n🔧 Укажите номер или название колонки для каждой роли:")

    column_roles = {
        "case_id": ask_column(columns, "Case ID (ID экземпляра процесса)"),
        "activity": ask_column(columns, "Activity (название действия/события)"),
        "timestamp": ask_column(columns, "Timestamp (временная метка)"),
    }

    print("\n✅ Маппинг колонок:")
    print(f"   Case ID   → {column_roles['case_id']}")
    print(f"   Activity  → {column_roles['activity']}")
    print(f"   Timestamp → {column_roles['timestamp']}")

    return column_roles


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def generate_report(session_dir: str, df: pd.DataFrame, column_roles: dict):
    """Generates and saves a basic PM report."""
    rows = len(df)
    cols = len(df.columns)
    col_names = ", ".join(list(df.columns))

    report_content = (
        "# Process Mining Data Report\n\n"
        "## Dataset Overview\n"
        f"- **Rows**: {rows}\n"
        f"- **Columns**: {cols}\n"
        f"- **Column Names**: {col_names}\n\n"
        "## Column Roles (PM Mapping)\n"
        f"- **Case ID**: `{column_roles['case_id']}`\n"
        f"- **Activity**: `{column_roles['activity']}`\n"
        f"- **Timestamp**: `{column_roles['timestamp']}`\n"
    )

    report_path = os.path.join(session_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n📄 Report saved to {report_path}")
    print("\n--- REPORT PREVIEW ---")
    print(report_content)
    print("----------------------")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AutoPM Agent")
    parser.add_argument("--file", type=str, help="Path to event log file")
    args = parser.parse_args()

    print("🤖 AutoPM Agent")

    # 1. Get file path (FIRST!)
    file_path = args.file
    if not file_path:
        print("Please provide a file using --file argument or input below.")
        file_path = input("Enter path to dataset file: ").strip().strip('"')

    # 2. Check for existing session (IMMEDIATELY)
    session_dir = get_session_dir(file_path)
    existing = load_session(session_dir)

    if existing:
        df, column_roles, meta = existing
        print(f"\n✅ Загружена предыдущая сессия от {meta['saved_at']}")
        print(f"   Файл: {meta['source_file']}")
        print(f"   Строк: {meta['rows']}, Столбцов: {meta['columns']}")
        print(f"   Case ID → {column_roles['case_id']}, "
              f"Activity → {column_roles['activity']}, "
              f"Timestamp → {column_roles['timestamp']}")

    # 3. Init LLM (needed for formatting and chat)
    print("🤖 Connecting to LLM...")
    try:
        llm_client = LLMClient()
    except Exception as e:
        print(f"❌ Error initializing LLM Client: {e}")
        return

    if not existing:
        # 4. Load raw data
        print(f"\n📂 Loading data from {file_path}...")
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return

        print(f"✅ Data loaded. ({len(df)} rows, {len(df.columns)} columns)")

        # 5. Column Mapping
        column_roles = map_columns(df)

        # 6. Format Data
        print("\n🧹 Formatting data types using DataFormatter...")
        try:
            formatter = DataFormatterAgent(df, llm_client)
            df = formatter.run()
            print("✅ Data formatting complete.")
        except Exception as e:
            print(f"⚠️ Formatting failed: {e}. Proceeding with raw data.")

        # 7. Save session
        save_session(session_dir, df, column_roles, file_path)

    # 8. Generate report
    generate_report(session_dir, df, column_roles)

    # 9. Chat Loop
    rows = len(df)
    cols = len(df.columns)
    col_names = ", ".join(list(df.columns))

    context_str = (
        f"Rows: {rows}, Columns: {cols}.\n"
        f"Column Names: {col_names}.\n"
        f"PM Roles: Case ID = '{column_roles['case_id']}', "
        f"Activity = '{column_roles['activity']}', "
        f"Timestamp = '{column_roles['timestamp']}'."
    )

    # Load chat history
    history_path = os.path.join(session_dir, "chat_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
        print(f"\n📜 Loaded {len(chat_history)} previous messages.")
    else:
        chat_history = []

    print("\n💬 Chat with your data! (Type 'exit' to quit)")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q']:
                break

            if not user_input:
                continue

            print("🤖 Thinking...")
            response = llm_client.simple_chat(user_input, context_str, chat_history)
            print(f"🤖 {response}")

            # Save to history
            chat_history.append({"role": "user", "text": user_input})
            chat_history.append({"role": "assistant", "text": response})

            # Persist history to disk
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    main()
