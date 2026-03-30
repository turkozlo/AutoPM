import argparse
import json
import pandas as pd
import sys
import os
import warnings
import pm4py
import csv
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
from pm_agent.agents.deviation_detector import DeviationDetectorAgent
from pm_agent.safe_executor import execute_pandas_code, validate_code_syntax, get_df_info_for_llm
from pm_agent.rag_manager import RAGManager
from pm_agent.config import RAG_DOC_DIR, RAG_MODEL_PATH


# ---------------------------------------------------------------------------
#  Console helpers
# ---------------------------------------------------------------------------

def robust_input(prompt: str) -> str:
    """Reads input safely, handling encoding issues in non-UTF-8 terminals."""
    import sys
    try:
        # Try to reconfigure stdin if possible (Python 3.7+)
        if hasattr(sys.stdin, 'reconfigure'):
            try:
                sys.stdin.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass
        return input(prompt)
    except UnicodeDecodeError:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        # Fallback: read raw bytes and decode with replacement
        raw_data = sys.stdin.buffer.readline()
        # Try common encodings, then fallback to 'replace'
        for enc in ['utf-8', 'cp1251', 'latin-1']:
            try:
                return raw_data.decode(enc).strip()
            except UnicodeError:
                continue
        return raw_data.decode('utf-8', errors='replace').strip()


# ---------------------------------------------------------------------------
#  Session helpers
# ---------------------------------------------------------------------------

def get_session_dir(file_path: str) -> str:
    """Returns the session directory for a given dataset file."""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join("reports", basename)


def save_session(session_dir: str, df: pd.DataFrame, column_roles: dict, file_path: str, findings_summary: str):
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
        "findings_summary": findings_summary
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

    # Standard XES/pm4py column names
    defaults = {
        "case_id": "case:concept:name" if "case:concept:name" in columns else (columns[0] if columns else ""),
        "activity": "concept:name" if "concept:name" in columns else (columns[1] if len(columns) > 1 else ""),
        "timestamp": "time:timestamp" if "time:timestamp" in columns else (columns[2] if len(columns) > 2 else "")
    }

    print("\n📋 Колонки в датасете:")
    print("-" * 40)
    for i, col in enumerate(columns, 1):
        sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        if len(sample) > 30:
            sample = sample[:30] + "..."
        print(f"  {i:3d}. {col:<30s}  (пример: {sample})")
    print("-" * 40)

    print("\n🔧 Проверьте или укажите колонки для ролей (Enter для значения по умолчанию):")
    
    col_case = robust_input(f"  Case ID [{defaults['case_id']}]: ").strip() or defaults['case_id']
    col_act = robust_input(f"  Activity [{defaults['activity']}]: ").strip() or defaults['activity']
    col_ts = robust_input(f"  Timestamp [{defaults['timestamp']}]: ").strip() or defaults['timestamp']

    # Simple validation/lookup
    def resolve_col(val, cols):
        if val.isdigit() and 1 <= int(val) <= len(cols): return cols[int(val)-1]
        return val if val in cols else val # Fallback to literal if not found (ask_column would be better but let's keep it simple for now)

    column_roles = {
        "case_id": resolve_col(col_case, columns),
        "activity": resolve_col(col_act, columns),
        "timestamp": resolve_col(col_ts, columns),
    }

    print("\n✅ Маппинг колонок:")
    print(f"   Case ID   → {column_roles['case_id']}")
    print(f"   Activity  → {column_roles['activity']}")
    print(f"   Timestamp → {column_roles['timestamp']}")

    return column_roles


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def generate_report(session_dir: str, df: pd.DataFrame, column_roles: dict, findings_summary: str):
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
        f"- **Timestamp**: `{column_roles['timestamp']}`\n\n"
        f"{findings_summary}\n"
    )

    report_path = os.path.join(session_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n📄 Report saved to {report_path}")
    print("\n--- REPORT PREVIEW ---")
    print(report_content)
    print("----------------------")


def load_csv_robustly(file_path: str) -> pd.DataFrame:
    """Try to load CSV with multiple encodings and automatic delimiter detection."""
    # Common encodings for regional data (UTF-8, Windows-1251, Western)
    encodings = ['utf-8', 'cp1251', 'latin-1', 'utf-8-sig']
    last_err = None
    
    for enc in encodings:
        try:
            # sep=None and engine='python' enable automatic delimiter detection
            return pd.read_csv(file_path, sep=None, engine='python', encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
        except Exception as e:
            # Other errors (file not found, etc.) should be raised immediately
            raise e
            
    if last_err:
        raise last_err
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="AutoPM Agent")
    parser.add_argument("--file", type=str, help="Path to event log file")
    args = parser.parse_args()

    print("AutoPM Agent")
    print(f"DEBUG: pandas dtype_backend = {pd.get_option('mode.dtype_backend')}")

    # 1. Get file path (FIRST!)
    file_path = args.file
    if not file_path:
        print("Please provide a file using --file argument or input below.")
        file_path = robust_input("Enter path to dataset file: ").strip().strip('"')

    # 2. Check for existing session (IMMEDIATELY)
    session_dir = get_session_dir(file_path)
    existing = load_session(session_dir)
    findings_summary = ""

    if existing:
        df, column_roles, meta = existing
        findings_summary = meta.get("findings_summary", "")
        print(f"\nLoaded previous session from {meta['saved_at']}")
        print(f"   Файл: {meta['source_file']}")
        print(f"   Строк: {meta['rows']}, Столбцов: {meta['columns']}")
        print(f"   Case ID → {column_roles['case_id']}, "
              f"Activity → {column_roles['activity']}, "
              f"Timestamp → {column_roles['timestamp']}")

    # 3. Init RAG and LLM
    print("Initializing RAG Knowledge Base...")
    rag_manager = None
    if os.path.exists(RAG_MODEL_PATH):
        try:
            rag_manager = RAGManager(RAG_DOC_DIR, RAG_MODEL_PATH)
        except Exception as e:
            print(f"⚠️ RAG Initialization failed: {e}")
    else:
        print(f"ℹ️ RAG model not found at {RAG_MODEL_PATH}, skipping RAG.")

    print("Connecting to LLM...")
    try:
        llm_client = LLMClient(rag_manager=rag_manager)
    except Exception as e:
        print(f"❌ Error initializing LLM Client: {e}")
        return

    if not existing:
        # 4. Load raw data
        print(f"\nLoading data from {file_path}...")
        try:
            if file_path.endswith('.csv'):
                df = load_csv_robustly(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.xes') or file_path.endswith('.xes.gz'):
                print("Using pm4py to read XES...")
                df = pm4py.read_xes(file_path)
            else:
                # Default attempt as CSV if extension is unknown
                try:
                    df = load_csv_robustly(file_path)
                except Exception:
                    df = pd.read_csv(file_path) # Final fallback
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return

        print(f"Data loaded. ({len(df)} rows, {len(df.columns)} columns)")

        # 5. Column Mapping
        column_roles = map_columns(df)

        # 6. Format Data (Force correct types BEFORE analysis)
        print("\nFormatting data types using DataFormatter...")
        try:
            formatter = DataFormatterAgent(df, llm_client)
            df = formatter.run()
            print("Data formatting complete.")
        except Exception as e:
            print(f"⚠️ Formatting failed: {e}. Proceeding with original data.")

        # 7. Deviation Detection
        print("\nRunning Deviation Detection...")
        try:
            detector = DeviationDetectorAgent(
                case_col=column_roles['case_id'],
                activity_col=column_roles['activity'],
                timestamp_col=column_roles['timestamp']
            )
            # 1. Preprocess (renames columns to pm4py standard)
            df, quality = detector.preprocess_event_log(df)
            
            if quality['warnings']:
                print("\nData Quality Warnings:")
                for w in quality['warnings']: print(f"  - {w}")
            
            # 2. Run analyzers on cleaned data
            detector.run_analysis(df)
            findings_summary = detector.get_summary_text()
            print("Deviation detection complete.")
        except Exception as e:
            import traceback
            print(f"Deviation detection failed: {e}")
            traceback.print_exc()
            findings_summary = "Deviation detection was skipped or failed due to an error."

        # 8. Save session
        save_session(session_dir, df, column_roles, file_path, findings_summary)

    # 9. Generate report
    generate_report(session_dir, df, column_roles, findings_summary)

    # 10. Chat Loop with Code Interpreter
    rows = len(df)
    cols = len(df.columns)
    col_names = ", ".join(list(df.columns))

    context_str = (
        f"Rows: {rows}, Columns: {cols}.\n"
        f"Column Names: {col_names}.\n"
        f"PM Roles: Case ID = '{column_roles['case_id']}', "
        f"Activity = '{column_roles['activity']}', "
        f"Timestamp = '{column_roles['timestamp']}'.\n"
        f"Findings from Deviation analysis:\n{findings_summary}"
    )

    # Load chat history
    history_path = os.path.join(session_dir, "chat_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
        print(f"\n📜 Loaded {len(chat_history)} previous messages.")
    else:
        chat_history = []

    # Prepare DataFrame info for Code Interpreter
    df_info = get_df_info_for_llm(df)

    # Load session errors (Global memory of past failures)
    errors_path = os.path.join(session_dir, "session_errors.json")
    if os.path.exists(errors_path):
        with open(errors_path, "r", encoding="utf-8") as f:
            session_errors = json.load(f)
    else:
        session_errors = []

    print("\n💬 Чат с данными! (Введите 'exit' для выхода)")
    print("   Агент может генерировать и выполнять Python-код для ответов на ваши вопросы.\n")

    while True:
        try:
            user_input = robust_input("\n👤 Вы: ").strip()
            if user_input.lower() in ['exit', 'quit', 'q', 'выход']:
                print("🏁 Завершение работы. До свидания!")
                break

            if not user_input:
                continue

            # --- Step 1: Smart Router ---
            print("🤔 Думаю...")
            resp = llm_client.simple_chat(user_input, context_str, chat_history)
            needs_code = resp.get("needs_code", False)
            answer_text = resp.get("answer")

            # Keyword fallback: force code if question looks computational
            calc_keywords = ["сколько", "посчитай", "вычисли", "найди", "покажи",
                             "среднее", "медиана", "топ", "процент", "сумма"]
            if not needs_code and any(kw in user_input.lower() for kw in calc_keywords):
                needs_code = True
                answer_text = None

            # --- Step 2: Code Interpreter (if needed) ---
            if needs_code:
                print("🧠 Запуск Code Interpreter...")
                MAX_CODE_ATTEMPTS = 3
                previous_error = ""

                for attempt in range(MAX_CODE_ATTEMPTS):
                    # Generate code (passing current previous_error and global session_errors)
                    all_errors = ""
                    if session_errors:
                        all_errors += "РАНЕЕ В ЭТОЙ СЕССИИ БЫЛИ ОШИБКИ (избегай их):\n" + "\n".join(session_errors[-5:]) + "\n\n"
                    if previous_error:
                        all_errors += f"ОШИБКА ТЕКУЩЕЙ ПОПЫТКИ:\n{previous_error}"

                    code_response = llm_client.generate_pandas_code(
                        user_input, df_info, all_errors
                    )
                    thought = code_response.get("thought", "")
                    code = code_response.get("code", "")

                    print(f"💭 Мысль: {thought}")
                    print(f"📝 Код:\n```python\n{code}\n```")

                    # Validate syntax
                    validation = validate_code_syntax(code)
                    if not validation["success"]:
                        print(f"⚠️ Синтаксическая ошибка: {validation['error']}")
                        previous_error = validation["error"]
                        if attempt == MAX_CODE_ATTEMPTS - 1:
                            answer_text = f"Не удалось сгенерировать корректный код. Ошибка: {previous_error}"
                        continue

                    # User confirmation
                    print("\n⚠️ ВНИМАНИЕ: Агент сгенерировал код для выполнения.")
                    confirm = robust_input("Нажмите Enter для выполнения или любой текст для отмены: ")
                    if confirm:
                        print("🚫 Выполнение отменено пользователем.")
                        answer_text = "Выполнение кода было отменено пользователем."
                        break

                    # Execute in sandbox
                    exec_result = execute_pandas_code(code, df)

                    if exec_result["success"]:
                        # Display path if it's a plot
                        if str(exec_result['result']).endswith('.png'):
                            print(f"🎨 Сгенерирован график: {exec_result['result']}")
                        else:
                            print(f"✅ Результат: {exec_result['result']}")

                        # Verify result (Passing history)
                        print("🔎 Самопроверка результата...")
                        verification = llm_client.verify_result(
                            user_input, exec_result["result"], chat_history
                        )

                        if verification.get("is_valid", True) is False:
                            print(f"🤔 Самопроверка: {verification.get('critique')}")
                            previous_error = (
                                f"Результат некорректен: {verification.get('critique')}. "
                                f"{verification.get('suggestion')}"
                            )
                            if attempt == MAX_CODE_ATTEMPTS - 1:
                                answer_text = "Не удалось получить точный результат после нескольких попыток."
                                break
                            continue  # Retry

                        # Interpret result
                        interp = llm_client.interpret_code_result(
                            user_input, exec_result["result"], exec_result["result_type"]
                        )
                        answer_text = interp.get("answer", str(exec_result["result"]))
                        break
                    else:
                        print(f"❌ Ошибка выполнения: {exec_result['error']}")
                        previous_error = exec_result["error"]
                        # Save to global session errors
                        if previous_error not in session_errors:
                            session_errors.append(f"Q: {user_input} | ERR: {previous_error}")
                        
                        if attempt == MAX_CODE_ATTEMPTS - 1:
                            answer_text = f"Код не удалось выполнить. Ошибка: {exec_result['error']}"

            # --- Step 3: Show answer ---
            if answer_text:
                print(f"\n🤖 {answer_text}")
            else:
                answer_text = "Не удалось получить ответ."
                print(f"\n🤖 {answer_text}")

            # Save to history
            chat_history.append({"role": "user", "text": user_input})
            chat_history.append({"role": "assistant", "text": answer_text})

            # Persist history to disk
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)
            
            # Persist errors to disk
            with open(errors_path, "w", encoding="utf-8") as f:
                json.dump(session_errors[-20:], f, ensure_ascii=False, indent=2)

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break


if __name__ == "__main__":
    main()
