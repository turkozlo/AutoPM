import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pm_agent.safe_executor import (
    execute_pandas_code,
    get_df_info_for_llm,
    validate_code_syntax,
)
from pm_agent.rag import RAGManager
from pm_agent.llm import LLMClient
from pm_agent.data_processor import DataProcessor  # Used for loading only
from pm_agent.config import MISTRAL_API_KEY  # noqa: F401
from pm_agent.chat_tools import execute_tool, get_tools_description
from pm_agent.agents.visualization import VisualizationAgent
from pm_agent.agents.report import ReportAgent
from pm_agent.agents.profiling import DataProfilingAgent
from pm_agent.agents.discovery import ProcessDiscoveryAgent
from pm_agent.agents.cleaning import DataCleaningAgent
from pm_agent.agents.analysis import ProcessAnalysisAgent
import glob
import argparse
import datetime
import json
import os
import time

import pandas as pd

# Force UTF-8 for console output
sys.stdout.reconfigure(encoding="utf-8")


# Import Agents


def safe_input(prompt: str = "") -> str:
    """
    Safely read user input with Unicode error handling.
    Handles UnicodeDecodeError that can occur when special keys (like backspace)
    produce invalid UTF-8 byte sequences in certain terminal environments.
    Invalid bytes are automatically filtered out without requiring re-entry.
    """
    try:
        # Read raw bytes from stdin to handle potential encoding issues
        sys.stdout.write(prompt)
        sys.stdout.flush()

        if hasattr(sys.stdin, "buffer"):
            # Read raw bytes and decode with error handling
            raw_bytes = sys.stdin.buffer.readline()
            # Decode with 'ignore' to silently drop invalid bytes
            user_input = raw_bytes.decode("utf-8", errors="ignore").rstrip("\n\r")
        else:
            # Fallback for environments without buffer access
            user_input = input()

        return user_input.strip()
    except EOFError:
        # Handle Ctrl+D
        return "exit"
    except KeyboardInterrupt:
        # Handle Ctrl+C
        print()
        return "exit"
    except Exception:
        try:
            return input().encode("utf-8", errors="replace").decode("utf-8").strip()
        except Exception:
            return ""


def find_latest_session(base_filename: str) -> str | None:
    """Finds the latest session directory for the given filename."""
    search_pattern = os.path.join("reports", f"{base_filename}_*")
    dirs = glob.glob(search_pattern)
    if not dirs:
        return None
    # Sort by creation time (name contains timestamp, so lexicographical sort works too if format is YYYYMMDD_HHMMSS)
    dirs.sort(reverse=True)
    return dirs[0]


def run_tool_wrapper(tool_name: str, agent_func, llm: LLMClient, **kwargs):
    """
    Executes a step (tool) with infinite retry logic (capped at 15) using feedback.
    """
    result = None
    feedback = ""
    result_str = ""
    MAX_SAFE_RETRIES = 15

    for attempt in range(1, MAX_SAFE_RETRIES + 1):
        try:
            # Try passing feedback if available
            if feedback:
                print(f"   🔄 Передача критики агенту: {feedback[:100]}...")
                try:
                    # Optimistically pass feedback
                    result = agent_func(feedback=feedback, **kwargs)
                except TypeError:
                    # Fallback for agents that don't support feedback yet
                    result = agent_func(**kwargs)
            else:
                result = agent_func(**kwargs)

            # Ensure result_str is string for Judge
            if isinstance(result, tuple):
                result_str = result[0]
            else:
                result_str = (
                    result
                    if isinstance(result, str)
                    else json.dumps(result, indent=2, ensure_ascii=False)
                )

        except Exception as e:
            result_str = str(f"Ошибка выполнения: {e}")
            print(result_str)

        # Judge the result
        judge_res = llm.judge_step(tool_name, f"Попытка {attempt}", result_str)
        passed = judge_res.get("passed", True)
        critique = judge_res.get("critique", "Нет замечаний")
        score = judge_res.get("score", 5)

        # Display thoughts (briefly)
        try:
            res_data = json.loads(result_str)
            if isinstance(res_data, dict) and "thoughts" in res_data:
                print(f"   💡 Thoughts: {res_data['thoughts']}")
        except (json.JSONDecodeError, TypeError):
            pass

        print(
            f"   >>> Судья (Попытка {attempt}/{MAX_SAFE_RETRIES}): {'ПРИНЯТО' if passed else 'ОТКЛОНЕНО'} (Оценка: {score})."
        )
        if not passed:
            print(f"   📝 Критика: {critique}")

        if passed or attempt == MAX_SAFE_RETRIES:
            if attempt == MAX_SAFE_RETRIES and not passed:
                print(
                    f"   ⚠️ Внимание: Достигнут лимит попыток ({MAX_SAFE_RETRIES}). Принимаем результат 'как есть'."
                )
            return result

        feedback = critique

    return result_str


def print_used_tools(used_tools: set):
    """Prints a summary of all tools used during the session."""
    if not used_tools:
        return
    print("\n=========================================")
    print("🛠️ ИСПОЛЬЗОВАННЫЕ ИНСТРУМЕНТЫ")
    for i, tool in enumerate(sorted(used_tools), 1):
        print(f"{i}. {tool}")
    print("=========================================\n")


def main():
    start_time = time.time()
    used_tools = set()
    parser = argparse.ArgumentParser(description="Process Mining AI Agent (ReAct)")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to the log file (CSV)"
    )
    parser.add_argument(
        "--rag-file", type=str, help="Path to the Excel file for RAG (optional)"
    )
    args = parser.parse_args()

    # Setup Output
    input_filename = Path(args.file).stem

    # Check for previous session
    latest_session = find_latest_session(input_filename)
    resume_mode = False
    output_dir = ""

    # Init RAG
    rag_manager = None
    if args.rag_file:
        rag_manager = RAGManager()
        rag_manager.load_excel(args.rag_file)

    # Init Core
    llm = LLMClient(rag_manager=rag_manager)
    artifacts = {}
    memory = ""
    current_df = None
    chat_history = []

    # Default Knowledge Base Content
    default_kb_content = (
        "# Knowledge Base & Glossary\n\n"
        "## Definitions (User Defaults)\n"
        "- **Экземпляр (Case)**: Это контейнер для набора событий. В одном экземпляре может быть много процессов (активностей).\n"
        "- **Процесс (Activity)**: Какая-то конкретная активность/действие. Одна и та же активность может встречаться в разных экземплярах.\n\n"
        "## User Insights\n"
    )

    if latest_session:
        print(f"\n🔍 Найдена предыдущая сессия: {latest_session}")
        choice = safe_input("Возобновить работу с ней? (y/n): ").lower()
        if choice == "y":
            resume_mode = True
            output_dir = latest_session
            print(f"📂 Загрузка состояния из {output_dir}...")

            # Load Memory
            try:
                with open(
                    os.path.join(output_dir, "memory.md"), "r", encoding="utf-8"
                ) as f:
                    memory = f.read()
                print("✅ Память загружена.")
            except FileNotFoundError:
                print("⚠️ memory.md не найден. Начинаем с пустой памяти.")
                memory = "Resumed session. Memory file missing."

            # Load Final Report (for Context)
            try:
                with open(
                    os.path.join(output_dir, "final_report.md"), "r", encoding="utf-8"
                ) as f:
                    final_report_content = f.read()
                print("✅ Финальный отчет загружен.")
            except FileNotFoundError:
                final_report_content = "Report missing."

            # Load Chat History
            chat_json_path = os.path.join(output_dir, "chat_history.json")
            if os.path.exists(chat_json_path):
                try:
                    with open(chat_json_path, "r", encoding="utf-8") as f:
                        chat_history = json.load(f)
                    print(f"✅ История чата загружена ({len(chat_history)} сообщений).")
                except Exception:
                    print("⚠️ Ошибка загрузки чата.")

            # Check/Create Knowledge Base
            kb_path = os.path.join(output_dir, "knowledge_base.md")
            if not os.path.exists(kb_path):
                print("⚠️ База знаний не найдена, создаю новую...")
                with open(kb_path, "w", encoding="utf-8") as f:
                    f.write(default_kb_content)

    if not resume_mode:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("reports", f"{input_filename}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Результаты: {output_dir}")
        memory = "Session started. Data loaded successfully. No analysis steps performed yet."

        # 1. Load Data
        loader = DataProcessor(args.file)
        success, msg = loader.load_data()
        if not success:
            print(f"Critical Error: {msg}")
            print_used_tools(used_tools)
            return
        df = loader.df
        print(f"Данные загружены. Строк: {len(df)}")
        current_df = df.copy()  # Working copy

        # Initialize Knowledge Base for new session
        kb_path = os.path.join(output_dir, "knowledge_base.md")
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write(default_kb_content)

    else:
        # Load DataFrame for resume mode (needed for tools)
        df_path = os.path.join(output_dir, "dataframe.pkl")
        if os.path.exists(df_path):
            current_df = pd.read_pickle(df_path)
            print(f"✅ DataFrame загружен ({len(current_df)} строк).")
        else:
            current_df = None
            print("⚠️ DataFrame не найден. Инструменты анализа недоступны.")

    # Tools definition
    tools_desc = """
    1. Data Profiling: Анализирует колонки, типы данных, пропуски. Определяет кандидатов на Case ID, Activity, Timestamp.
    2. Data Cleaning: Очищает данные (пропуски, типы) на основе профилирования. ТРЕБУЕТ Data Profiling.
    3. Process Discovery: Строит DFG граф процесса и Mermaid схему. ТРЕБУЕТ очищенных данных.
    4. Visualization: Строит графики распределения событий и времени.
    5. Process Analysis: Считает производительность, ищет узкие места (bottlenecks). ТРЕБУЕТ Process Discovery (PM columns).
    6. Reporting: Генерирует финальный отчет Markdown. ТРЕБУЕТ выполнения основных шагов анализа.
    7. Finish: Завершает работу. Использовать ТОЛЬКО после Reporting.
    """

    MAX_STEPS = 10

    # --- Session Loop (Global Retry) ---
    # --- Session Loop (Global Retry) ---
    MAX_SESSION_RETRIES = 2  # 1 initial + 1 retry
    session_attempt = 0
    # If resuming, we assume the previous session was passed or user wants to chat anyway
    global_passed = True if resume_mode else False

    # Only run analysis loop if NOT resuming
    if not resume_mode:
        while session_attempt < MAX_SESSION_RETRIES and not global_passed:
            session_attempt += 1
            print("=========================================")
            print(f"🚀 ЗАПУСК СЕССИИ (Попытка {session_attempt}/{MAX_SESSION_RETRIES})")
            print("=========================================\n")

            step_count = 0
            final_report_content = ""
            cumulative_outputs = []

            while step_count < MAX_STEPS:
                step_count += 1
                print(f"\n--- Раунд {step_count} (Сессия {session_attempt}) ---")

                # 1. Decide (using Memory)
                decision = llm.decide_next_step(memory, tools_desc)

                thought = decision.get("thought", "No thought")
                tool_name = decision.get("tool_name", "Unknown")

                print(f"💭 МЫСЛЬ АГЕНТА: {thought}")
                print(f"👉 ВЫБОР ИНСТРУМЕНТА: {tool_name}")

                if tool_name == "Finish":
                    used_tools.add(tool_name)
                    print("🏁 Агент запросил завершение работы.")
                    break

                used_tools.add(tool_name)
                # 2. Execute
                current_step_result_str = ""

                try:
                    if tool_name == "Data Profiling":
                        agent = DataProfilingAgent(current_df.copy(), llm)
                        res_json = run_tool_wrapper(tool_name, agent.run, llm)
                        artifacts["profiling"] = json.loads(res_json)
                        current_step_result_str = (
                            f"Data Profiling completed. Readiness: "
                            f"{artifacts['profiling'].get('process_mining_readiness', {}).get('level')}"
                        )

                    elif tool_name == "Data Cleaning":
                        if "profiling" not in artifacts:
                            raise ValueError("Requires Profiling first")
                        agent = DataCleaningAgent(current_df.copy(), llm)
                        # Cleaning returns tuple (report, new_df)
                        res = run_tool_wrapper(
                            tool_name,
                            agent.run,
                            llm,
                            profiling_report=artifacts["profiling"],
                        )

                        clean_report_json = ""
                        if isinstance(res, tuple):
                            clean_report_json, cleaned_df = res
                            current_df = cleaned_df  # Update working DF
                        else:
                            clean_report_json = res

                        artifacts["cleaning"] = json.loads(clean_report_json)
                        current_step_result_str = (
                            "Data Cleaning completed. DataFrame updated."
                        )

                    elif tool_name == "Visualization":
                        if "profiling" not in artifacts:
                            raise ValueError(
                                "Requires Profiling first for column detection"
                            )
                        agent = VisualizationAgent(current_df.copy(), llm)
                        res_json = run_tool_wrapper(
                            tool_name,
                            agent.run,
                            llm,
                            profiling_report=artifacts["profiling"],
                            output_dir=output_dir,
                        )
                        artifacts["visualization"] = json.loads(res_json)
                        current_step_result_str = (
                            "Visualization completed. 4 charts generated."
                        )

                    elif tool_name == "Process Discovery":
                        agent = ProcessDiscoveryAgent(current_df.copy(), llm)
                        # Pass output_dir
                        res_json = run_tool_wrapper(
                            tool_name,
                            agent.run,
                            llm,
                            pm_columns=None,
                            output_dir=output_dir,
                        )
                        artifacts["discovery"] = json.loads(res_json)
                        current_step_result_str = f"Process Discovery completed. Found {artifacts['discovery'].get('activities')} activities."

                    elif tool_name == "Process Analysis":
                        if "discovery" not in artifacts:
                            raise ValueError("Requires Discovery first (PM columns)")
                        confirmed_pm_cols = artifacts["discovery"].get("pm_columns")
                        agent = ProcessAnalysisAgent(current_df.copy(), llm)
                        res_json = run_tool_wrapper(
                            tool_name,
                            agent.run,
                            llm,
                            pm_columns=confirmed_pm_cols,
                            output_dir=output_dir,
                        )
                        artifacts["analysis"] = json.loads(res_json)
                        current_step_result_str = "Process Analysis completed. Performance metrics calculated."

                    elif tool_name == "Reporting":
                        if not artifacts:
                            current_step_result_str = (
                                "Reporting failed: No artifacts to report."
                            )
                        else:
                            agent = ReportAgent(llm)
                            final_report_content = run_tool_wrapper(
                                tool_name, agent.run, llm, artifacts=artifacts
                            )

                            report_path = os.path.join(output_dir, "final_report.md")
                            with open(report_path, "w", encoding="utf-8") as f:
                                f.write(final_report_content)
                            print(f"📄 Отчет сохранен: {report_path}")
                            current_step_result_str = f"Reporting completed. Final report saved to {report_path}."

                            # Save DataFrame for future resume/tools
                            df_path = os.path.join(output_dir, "dataframe.pkl")
                            current_df.to_pickle(df_path)
                            print(f"💾 DataFrame сохранен: {df_path}")

                    else:
                        current_step_result_str = f"Error: Unknown tool '{tool_name}'"
                        print("❌ Неизвестный инструмент.")

                    # Store raw output for cumulative judging
                    cumulative_outputs.append(
                        f"Step {step_count}: {tool_name}\nResult:\n{current_step_result_str}"
                    )

                except Exception as e:
                    err_msg = f"Error executing {tool_name}: {e}"
                    print(f"❌ {err_msg}")
                    current_step_result_str = err_msg

                # 3. Update Long-Term Memory
                print("   🧠 Обновление памяти...")
                memory = llm.update_memory(memory, tool_name, current_step_result_str)

                # Save memory trace
                with open(
                    os.path.join(output_dir, "memory.md"), "w", encoding="utf-8"
                ) as f:
                    f.write(memory)

                # 4. Global Cumulative Step Evaluation
                print("\n⚖️ ГЛОБАЛЬНАЯ ОЦЕНКА ПРОГРЕССА СЕССИИ...")
                cumulative_context = "\n---\n".join(cumulative_outputs)
                judge_verdict = llm.judge_cumulative_progress(
                    memory, len(artifacts), cumulative_context
                )
                global_passed = judge_verdict.get("passed", False)
                critique = judge_verdict.get("critique", "Нет замечаний")

                if not global_passed:
                    print(f"❌ ПРОГРЕСС СЕССИИ ПРИЗНАН НЕУСПЕШНЫМ. Критика: {critique}")
                    if session_attempt < MAX_SESSION_RETRIES:
                        print("🔄 ПЕРЕЗАПУСК СЕССИИ С УЧЕТОМ КРИТИКИ...")
                        # Inject critique into memory for next run
                        memory = f"Previous Session Failed.\nCritique: {critique}\nRestarting process...\nData loaded."
                        current_df = df.copy()  # Reset DF
                        artifacts = {}  # Reset artifacts
                        # Break out of step loop to restart session
                        break
                    else:
                        print("⛔ Все попытки сессии исчерпаны.")
                        # Even if failed, we break to exit the analysis phase
                        break
                else:
                    print("✅ Прогресс сессии одобрен Судьей.")

                if step_count >= MAX_STEPS:
                    print("⚠️ Достигнут лимит шагов агента внутри сессии.")
                    global_passed = True  # Allow moving to chat if we hit step limit but judge was ok with progress so far
                    break

    if global_passed:
        # Calculate and print total time
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)
        time_str = f"{minutes} мин {seconds} сек" if minutes > 0 else f"{seconds} сек"
        print(f"\n⏱️ Общее время выполнения: {time_str}")

        # --- Interactive QA Mode ---
        print("\n=========================================")
        print("💬 РЕЖИМ КОНСУЛЬТАЦИИ")
        print(
            "Теперь вы можете задать вопросы по обработанному датасету. Контекст диалога сохраняется."
        )
        print("Введите вопрос или 'exit' для выхода.")
        print("=========================================\n")

        # Ensure chat_history is loaded (if resumed) or empty
        # chat_history is initialized at start of main()

        while True:
            try:
                user_input = safe_input("\n👤 Ваш вопрос: ")
                if user_input.lower() in ["exit", "quit", "выход"]:
                    print("🏁 Завершение работы. До свидания!")
                    print_used_tools(used_tools)
                    break

                if not user_input:
                    continue

                print("🤖 Агент думает...")

                # Context Window Management (Sliding Window)
                MAX_CHAT_CONTEXT = 20  # Keep last 20 messages
                recent_history = (
                    chat_history[-MAX_CHAT_CONTEXT:]
                    if len(chat_history) > MAX_CHAT_CONTEXT
                    else chat_history
                )

                # Convert history to string
                history_str = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in recent_history]
                )

                # Read Knowledge Base
                kb_path = os.path.join(output_dir, "knowledge_base.md")
                knowledge_base_content = ""
                if os.path.exists(kb_path):
                    with open(kb_path, "r", encoding="utf-8") as f:
                        knowledge_base_content = f.read()

                # Get Answer with potential Tool Use
                tools_desc = get_tools_description() if current_df is not None else ""
                response_data = llm.answer_user_question(
                    memory,
                    final_report_content,
                    history_str,
                    user_input,
                    knowledge_base_content,
                    tools_desc,
                )

                answer_text = None

                # Check if tool use is requested
                tool_call = response_data.get("tool_call")
                needs_code = response_data.get("needs_code", False)

                if tool_call and current_df is not None:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    used_tools.add(tool_name)
                    print(
                        f"🔧 Вызов инструмента: {tool_name} (аргументы: {json.dumps(tool_args, ensure_ascii=False)})"
                    )

                    # PROACTIVE ROUTER: If agent asks for complex analysis, skip direct execution and go to code
                    if tool_name == "run_complex_analysis":
                        print(
                            "🚀 Агент выбрал 'run_complex_analysis' — сразу запускаем Code Interpreter для сложной задачи."
                        )
                        needs_code = True
                        tool_result = {"status": "skipped_for_code"}  # Dummy result
                    else:
                        tool_result = execute_tool(tool_name, tool_args, current_df)

                    if "error" in tool_result:
                        print(f"⚠️ Ошибка инструмента: {tool_result['error']}")
                        print("🔄 Переключаюсь на Code Interpreter...")
                        needs_code = True
                    elif needs_code:
                        # Skip verification for proactive router
                        pass
                    else:
                        tool_result_str = str(tool_result)
                        # Verify result
                        print("🔎 Самопроверка результата...")
                        verification = llm.verify_result(user_input, tool_result_str)

                        thought = verification.get("thought", "Нет объяснения.")
                        print(f"💭 Мысль агента: {thought}")

                        is_valid = verification.get("is_valid", True)

                        if is_valid is False:
                            print(
                                f"🤔 Самопроверка: ответ кажется некорректным. {verification.get('critique')}"
                            )
                            print(f"💡 Совет: {verification.get('suggestion')}")
                            print(
                                "🔄 Результат инструмента не устраивает. Переключаюсь на Code Interpreter для точного решения..."
                            )
                            needs_code = True
                            answer_text = (
                                None  # Reset answer so it drops through to CI block
                            )
                        else:
                            if is_valid == "partial":
                                print(
                                    f"⚠️ Ответ принят с оговорками (Partial Success): {verification.get('critique')}"
                                )

                            print(
                                f"📊 Результат инструмента: {json.dumps(tool_result, ensure_ascii=False, indent=2)}"
                            )
                            followup_data = llm.interpret_tool_result(
                                user_input, tool_result
                            )
                            answer_text = followup_data.get("answer", str(tool_result))

                            # DOUBLE SAFETY: If answer admits failure, force fallback
                            failure_triggers = [
                                "не вижу",
                                "нет нужных данных",
                                "вернул только",
                            ]
                            if any(
                                trigger in answer_text.lower()
                                for trigger in failure_triggers
                            ):
                                print(
                                    f"🚨 Интерпретатор обнаружил нехватку данных: '{answer_text}'"
                                )
                                print(
                                    "🔄 Переключаюсь на Code Interpreter для точного решения..."
                                )
                                needs_code = True
                                answer_text = None

                # Code Interpreter fallback (dynamic pandas execution)
                if (needs_code or answer_text is None) and current_df is not None:
                    # Check if this looks like a calculation question
                    calc_keywords = [
                        "сколько",
                        "какой процент",
                        "посчитай",
                        "вычисли",
                        "найди",
                        "покажи",
                        "подсчитай",
                        "среднее",
                        "медиана",
                        "топ",
                        "редкий",
                        "частый",
                    ]

                    # Force checks if explicit tool failure requested code
                    is_calc_question = (
                        any(kw in user_input.lower() for kw in calc_keywords)
                        or needs_code
                    )

                    if is_calc_question:
                        print("🧠 Запуск Code Interpreter...")
                        used_tools.add("Code Interpreter")
                        df_info = get_df_info_for_llm(current_df)

                        MAX_CODE_ATTEMPTS = 3
                        previous_error = ""

                        for attempt in range(MAX_CODE_ATTEMPTS):
                            # Generate code
                            context = {
                                "knowledge_base": knowledge_base_content,
                                "memory": memory,
                                "final_report": final_report_content,
                            }
                            code_response = llm.generate_pandas_code(
                                user_input, df_info, previous_error, context=context
                            )
                            thought = code_response.get("thought", "")
                            code = code_response.get("code", "")

                            print(f"💭 Мысль: {thought}")
                            print(f"📝 Код:\n```python\n{code}\n```")

                            # 1. Validate Code Syntax
                            validation = validate_code_syntax(code)
                            if not validation["success"]:
                                print(
                                    f"⚠️ Найдена синтаксическая ошибка: {validation['error']}"
                                )
                                previous_error = validation["error"]
                                if attempt == MAX_CODE_ATTEMPTS - 1:
                                    answer_text = f"Не удалось сгенерировать корректный код. Ошибка: {previous_error}"
                                continue

                            # 2. User Confirmation
                            print(
                                "\n⚠️ ВНИМАНИЕ: Агент сгенерировал код для выполнения."
                            )
                            confirm = safe_input(
                                "Нажмите Enter для выполнения или любой текст для отмены: "
                            )
                            if confirm:
                                print("🚫 Выполнение отменено пользователем.")
                                answer_text = (
                                    "Выполнение кода было отменено пользователем."
                                )
                                break

                            # 3. Execute code
                            exec_result = execute_pandas_code(code, current_df)

                            if exec_result["success"]:
                                print(f"✅ Результат: {exec_result['result']}")

                                # VERIFY Code Result
                                verification = llm.verify_result(
                                    user_input, exec_result["result"]
                                )
                                if verification.get("is_valid", True) is False:
                                    print(
                                        f"🤔 Самопроверка кода: {verification.get('critique')}"
                                    )
                                    previous_error = (
                                        f"Результат был получен, но он некорректен: {verification.get('critique')}. "
                                        f"{verification.get('suggestion')}"
                                    )
                                    # Fallback: Store result just in case we run out of retries
                                    if attempt == MAX_CODE_ATTEMPTS - 1:
                                        answer_text = "К сожалению, мне не удалось выполнить расчет точно с помощью кода. Я постараюсь ответить, основываясь на имеющихся у меня общих данных о процессе."
                                        break
                                    continue  # Retry loop

                                # Interpret result
                                interp = llm.interpret_code_result(
                                    user_input,
                                    exec_result["result"],
                                    exec_result["result_type"],
                                )
                                answer_text = interp.get(
                                    "answer", exec_result["result"]
                                )
                                break
                            else:
                                previous_error = exec_result["error"]
                                print(f"❌ Техническая ошибка (попытка {attempt + 1}/{MAX_CODE_ATTEMPTS}): {previous_error}")
                                if attempt == MAX_CODE_ATTEMPTS - 1:
                                    answer_text = "Простите, при выполнении расчетов возникла техническая сложность. Я постараюсь ответить на ваш вопрос без использования кода."

                # Fallback to direct answer
                if answer_text is None:
                    answer_text = response_data.get(
                        "answer", "Не могу ответить на этот вопрос."
                    )

                knowledge_update = response_data.get("knowledge_update")

                print(f"👉 Ответ:\n{answer_text}")

                # Process Knowledge Update
                if knowledge_update:
                    print(f"\n📝 Агент сохраняет новое знание: {knowledge_update}")
                    with open(kb_path, "a", encoding="utf-8") as f:
                        f.write(f"\n- **User Insight**: {knowledge_update}")
                    # In-memory update for this loop iteration
                    knowledge_base_content += (
                        f"\n- **User Insight**: {knowledge_update}"
                    )

                # Update History
                chat_history.append({"role": "User", "content": user_input})
                chat_history.append({"role": "Assistant", "content": answer_text})

                # Persist Chat History (JSON)
                with open(
                    os.path.join(output_dir, "chat_history.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(chat_history, f, ensure_ascii=False, indent=2)

                # Persist Chat Log (Markdown)
                with open(
                    os.path.join(output_dir, "chat_log.md"), "a", encoding="utf-8"
                ) as f:
                    f.write(
                        f"**User**: {user_input}\n\n**Assistant**: {answer_text}\n\n---\n\n"
                    )

            except KeyboardInterrupt:
                print("\n🏁 Завершение работы.")
                print_used_tools(used_tools)
                break
            except Exception as e:
                print(f"❌ Ошибка в чате: {e}")

    print_used_tools(used_tools)


if __name__ == "__main__":
    main()
