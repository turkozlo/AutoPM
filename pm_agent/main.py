import argparse
import sys
import json
import pandas as pd
import os
import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Force UTF-8 for console output
sys.stdout.reconfigure(encoding='utf-8')

from pm_agent.config import MISTRAL_API_KEY
from pm_agent.llm import LLMClient
from pm_agent.data_processor import DataProcessor # Used for loading only

# Import New Agents
from pm_agent.agents.profiling import DataProfilingAgent
from pm_agent.agents.cleaning import DataCleaningAgent
from pm_agent.agents.visualization import VisualizationAgent
from pm_agent.agents.discovery import ProcessDiscoveryAgent
from pm_agent.agents.analysis import ProcessAnalysisAgent
from pm_agent.agents.report import ReportAgent

def run_step_with_retry(step_name: str, agent_func, llm: LLMClient, max_retries: int = 3, **kwargs):
    """
    Executes a step with retry logic based on Judge's feedback.
    """
    feedback = ""
    for attempt in range(max_retries + 1):
        print(f"\n--- {step_name} (Попытка {attempt + 1}) ---")
        
        try:
            # Execute Agent
            result = agent_func(**kwargs)
            
            # Ensure result_str is string for Judge, handle tuple (report, df)
            if isinstance(result, tuple):
                result_str = result[0]
            else:
                result_str = result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False)
            
        except Exception as e:
            result = f"Ошибка выполнения: {e}"
            result_str = str(result)
            print(result_str)

        # Judge the result
        judge_res = llm.judge_step(step_name, f"Попытка {attempt + 1}", result_str)
        passed = judge_res.get("passed", True)
        critique = judge_res.get("critique", "Нет замечаний")
        score = judge_res.get("score", 5)
        
        # Display Thoughts and Functions if available
        try:
            res_data = json.loads(result_str)
            if isinstance(res_data, dict):
                thoughts = res_data.get("thoughts")
                funcs = res_data.get("applied_functions")
                err = res_data.get("error")
                debug = res_data.get("debug_info")
                
                if thoughts:
                    print(f"\n💡 МЫСЛИ АГЕНТА: {thoughts}")
                if funcs:
                    print(f"🛠️ ПРИМЕНЕННЫЕ ФУНКЦИИ: {', '.join(funcs)}")
                if err:
                    print(f"\n❌ ОШИБКА АГЕНТА: {err}")
                if debug:
                    print(f"🔍 DEBUG INFO: {json.dumps(debug, indent=2, ensure_ascii=False)}")
        except:
            pass

        print(f"\nСудья: {'ПРИНЯТО' if passed else 'ОТКЛОНЕНО'} (Оценка: {score})")
        print(f"Комментарий: {critique}")
        
        if passed:
            return result # Return original result (likely JSON string)
        
        feedback = critique
        
    print(f"Внимание: Шаг '{step_name}' не прошел проверку Судьи после {max_retries} попыток. Продолжаем с последним результатом.")
    return result

def main():
    parser = argparse.ArgumentParser(description="Process Mining AI Agent (Multi-Agent)")
    parser.add_argument("--file", type=str, required=True, help="Path to the log file (CSV)")
    args = parser.parse_args()

    # Setup Output Directory
    input_filename = Path(args.file).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("reports", f"{input_filename}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Запуск PM Агента (Multi-Agent) с файлом: {args.file}")
    print(f"Результаты будут сохранены в: {output_dir}")

    # Initialize Core
    llm = LLMClient()
    
    # 1. Load Data (Legacy Loader)
    print("\n--- Шаг 1: Загрузка данных ---")
    loader = DataProcessor(args.file)
    success, msg = loader.load_data()
    if not success:
        print(f"Критическая ошибка: {msg}")
        return
    print(msg)
    df = loader.df

    artifacts = {}

    # 2. Data Profiling
    profiling_agent = DataProfilingAgent(df)
    profile_json = run_step_with_retry("Data Profiling", profiling_agent.run, llm)
    artifacts['profiling'] = json.loads(profile_json)

    # 3. Data Cleaning
    cleaning_agent = DataCleaningAgent(df, llm)
    clean_res = run_step_with_retry("Data Cleaning", cleaning_agent.run, llm, profiling_report=artifacts['profiling'])
    
    # Handle tuple return (report, cleaned_df)
    if isinstance(clean_res, tuple):
        clean_report_json, df = clean_res
    else:
        clean_report_json = clean_res
        
    artifacts['cleaning'] = json.loads(clean_report_json)
    
    # 4. Visualization
    vis_agent = VisualizationAgent(df, llm)
    vis_report_json = run_step_with_retry("Visualization", vis_agent.run, llm, profiling_report=artifacts['profiling'], output_dir=output_dir)
    try:
        artifacts['visualization'] = json.loads(vis_report_json)
    except:
        artifacts['visualization'] = {"error": "Failed to parse visualization JSON", "raw": vis_report_json}

    # 5. Process Discovery
    discovery_agent = ProcessDiscoveryAgent(df, llm)
    
    # Pass output_dir
    discovery_json = run_step_with_retry("Process Discovery", discovery_agent.run, llm, pm_columns=None, output_dir=output_dir)
    artifacts['discovery'] = json.loads(discovery_json)

    # 6. Process Analysis
    analysis_agent = ProcessAnalysisAgent(df, llm)
    
    confirmed_pm_cols = artifacts['discovery'].get('pm_columns')
    if not confirmed_pm_cols:
        print("Ошибка: Не удалось определить столбцы PM на этапе Discovery.")
        return

    analysis_json = run_step_with_retry("Process Analysis", analysis_agent.run, llm, pm_columns=confirmed_pm_cols, output_dir=output_dir)
    try:
        artifacts['analysis'] = json.loads(analysis_json)
    except:
        artifacts['analysis'] = {"error": "Failed to parse analysis JSON", "raw": analysis_json}

    # 7. Reporting
    report_agent = ReportAgent(llm)
    final_report = run_step_with_retry("Reporting", report_agent.run, llm, artifacts=artifacts)
    
    print("\n--- ФИНАЛЬНЫЙ ОТЧЕТ ---")
    print(final_report)
    
    # Save report
    report_path = os.path.join(output_dir, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\nОтчет сохранен в '{report_path}'")

    print("\nАгент завершил выполнение.")

if __name__ == "__main__":
    main()
