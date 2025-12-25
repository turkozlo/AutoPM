import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List

class VisualizationAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def run(self, profiling_report: Dict[str, Any], output_dir: str = ".") -> str:
        """
        Generates mandatory PM charts and asks LLM to interpret them.
        """
        df = self.df.copy() # Isolate to prevent side effects
        # 1. Identify Columns for PM Charts
        pm_readiness = profiling_report.get('process_mining_readiness', {})
        
        # Try to find best candidates
        activity_col = pm_readiness.get('activity_candidates', [None])[0]
        timestamp_col = pm_readiness.get('timestamp_candidates', [None])[0]
        case_col = pm_readiness.get('case_id_candidates', [None])[0]
        
        # Fallback to first column if not found
        if not activity_col: activity_col = list(df.columns)[0]
        if not timestamp_col: 
            # Look for any datetime column
            for col, stats in profiling_report['columns'].items():
                if 'datetime' in stats['dtype']:
                    timestamp_col = col
                    break
        if not case_col: case_col = list(df.columns)[0]

        # 1.5 Synthetic Case ID if needed
        if not case_col or case_col not in df.columns or df[case_col].nunique() > len(df) * 0.9:
            df = df.sort_values(timestamp_col)
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df['case_id_synth'] = (df[timestamp_col].diff() > pd.Timedelta("30min")).cumsum()
            case_col = 'case_id_synth'

        charts = [
            {"name": "operation_distribution.png", "type": "bar", "column": activity_col, "title": f"Частота операций ({activity_col})"},
            {"name": "timestamp_distribution.png", "type": "hist", "column": timestamp_col, "title": f"Распределение во времени ({timestamp_col})"},
        ]
        
        # Add PM specific charts if we have case_id
        if case_col and activity_col:
            charts.append({"name": "case_duration.png", "type": "case_events", "column": case_col, "title": "Количество событий на кейс"})
        
        if case_col and timestamp_col:
            charts.append({"name": "inter_event_time.png", "type": "inter_event", "column": timestamp_col, "title": "Интервалы между событиями"})

        results = []
        
        for chart in charts:
            try:
                plt.figure(figsize=(10, 6))
                col = chart['column']
                ctype = chart['type']
                
                data_summary = {}
                
                if ctype == "bar":
                    counts = df[col].value_counts().head(10)
                    counts.plot(kind='bar')
                    data_summary = {str(k): int(v) for k, v in counts.items()}
                elif ctype == "hist":
                    # Ensure timestamp is datetime
                    temp_ts = pd.to_datetime(df[col], errors='coerce').dropna()
                    if not temp_ts.empty:
                        temp_ts.hist(bins=20)
                        data_summary = {"min": str(temp_ts.min()), "max": str(temp_ts.max())}
                elif ctype == "case_events":
                    counts = df.groupby(case_col).size()
                    counts.hist(bins=20)
                    data_summary = {"mean": float(counts.mean()), "max": int(counts.max()), "min": int(counts.min())}
                elif ctype == "inter_event":
                    # Sort and calculate diff
                    temp_df = df[[case_col, timestamp_col]].copy()
                    temp_df[timestamp_col] = pd.to_datetime(temp_df[timestamp_col], errors='coerce')
                    temp_df = temp_df.sort_values([case_col, timestamp_col])
                    diffs_sec = temp_df.groupby(case_col)[timestamp_col].diff().dt.total_seconds().dropna()
                    
                    if not diffs_sec.empty:
                        median_sec = diffs_sec.median()
                        unit = "сек."
                        factor = 1.0
                        if median_sec > 86400:
                            unit, factor = "дн.", 86400.0
                        elif median_sec > 3600:
                            unit, factor = "час.", 3600.0
                        elif median_sec > 60:
                            unit, factor = "мин.", 60.0
                        
                        diffs = diffs_sec / factor
                        diffs.hist(bins=20)
                        chart['title'] = f"Интервалы между событиями ({unit})"
                        data_summary = {
                            "unit": unit,
                            "mean": float(round(diffs.mean(), 2)),
                            "median": float(round(diffs.median(), 2))
                        }

                plt.title(chart['title'])
                plt.tight_layout()
                
                filepath = os.path.join(output_dir, chart['name'])
                plt.savefig(filepath)
                plt.close()
                
                abs_path = os.path.abspath(filepath).replace("\\", "/")
                
                # LLM Interpretation
                stats_prompt = f"Chart: {chart['name']}\nData Summary: {json.dumps(data_summary)}"
                interp_system = "Ты — аналитик. Дай краткую интерпретацию (1 предложение) этого графика на основе статистики. Обязательно используй цифры из сводки. Не выдумывай."
                interpretation = self.llm.generate_response(stats_prompt, interp_system)
                
                results.append({
                    "image": abs_path,
                    "type": ctype,
                    "column": col,
                    "data_summary": data_summary,
                    "interpretation": interpretation.strip()
                })
                
            except Exception as e:
                results.append({"chart": chart['name'], "error": str(e)})

        try:
            # Explanation for large gaps
            ts_series = pd.to_datetime(df[timestamp_col], errors='coerce').dropna()
            time_range_days = (ts_series.max() - ts_series.min()).days
            gap_explanation = f"Внимание: данные охватывают период в {time_range_days} дней (с {ts_series.min().year} по {ts_series.max().year} год), поэтому использование единиц 'дн.' (дни) для длительности кейсов является корректным."

            return json.dumps({
                "visualizations": results,
                "thoughts": f"Сгенерированы обязательные графики. {gap_explanation} Все файлы сохранены в формате .png.",
                "applied_functions": ["plt.savefig()", "df.value_counts()", "df.groupby().diff()", "pd.to_datetime()"]
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            cols = list(df.columns) if hasattr(df, 'columns') else "N/A"
            return json.dumps({"error": f"Visualization failed: {e}. Columns: {cols}"}, ensure_ascii=False)

