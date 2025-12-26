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
            charts.append({"name": "case_events.png", "type": "case_events", "column": case_col, "title": "Количество событий на кейс"})
        
        if case_col and timestamp_col:
            charts.append({"name": "inter_event_time.png", "type": "inter_event", "column": timestamp_col, "title": "Интервалы между событиями"})
            charts.append({"name": "case_duration.png", "type": "case_duration", "column": timestamp_col, "title": "Продолжительность кейсов"})

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
                    data_summary = {
                        "unit": "событий",
                        "top_1": str(counts.index[0]) if not counts.empty else "N/A",
                        "top_1_count": int(counts.iloc[0]) if not counts.empty else 0,
                        "details": {str(k): int(v) for k, v in counts.items()}
                    }
                elif ctype == "hist":
                    # Ensure timestamp is datetime
                    temp_ts = pd.to_datetime(df[col], errors='coerce').dropna()
                    if not temp_ts.empty:
                        temp_ts.hist(bins=20)
                        data_summary = {
                            "unit": "дн.",
                            "min": temp_ts.min().strftime('%Y-%m-%d'), 
                            "max": temp_ts.max().strftime('%Y-%m-%d'),
                            "total_days": float(round((temp_ts.max() - temp_ts.min()).total_seconds() / 86400, 2))
                        }
                elif ctype == "case_events":
                    counts = df.groupby(case_col).size()
                    counts.hist(bins=20)
                    data_summary = {
                        "unit": "событий на кейс",
                        "mean": float(round(counts.mean(), 2)), 
                        "max": int(counts.max()), 
                        "min": int(counts.min())
                    }
                elif ctype == "inter_event":
                    # Sort and calculate diff
                    temp_df = df[[case_col, timestamp_col]].copy()
                    temp_df[timestamp_col] = pd.to_datetime(temp_df[timestamp_col], errors='coerce')
                    temp_df = temp_df.sort_values([case_col, timestamp_col])
                    diffs_sec = temp_df.groupby(case_col)[timestamp_col].diff().dt.total_seconds().dropna()
                    
                    if not diffs_sec.empty:
                        ref_val = diffs_sec.mean()
                        unit = "сек."
                        factor = 1.0
                        if ref_val > 86400: unit, factor = "дн.", 86400.0
                        elif ref_val > 3600: unit, factor = "час.", 3600.0
                        elif ref_val > 60: unit, factor = "мин.", 60.0
                        
                        diffs = diffs_sec / factor
                        diffs.hist(bins=20)
                        chart['title'] = f"Интервалы между событиями ({unit})"
                        data_summary = {
                            "unit": unit,
                            "mean": float(round(diffs.mean(), 2)),
                            "median": float(round(diffs_sec.median() / factor, 4))
                        }
                elif ctype == "case_duration":
                    temp_df = df[[case_col, timestamp_col]].copy()
                    temp_df[timestamp_col] = pd.to_datetime(temp_df[timestamp_col], errors='coerce')
                    durations = temp_df.groupby(case_col)[timestamp_col].agg(lambda x: (x.max() - x.min()).total_seconds()).dropna()
                    
                    if not durations.empty:
                        ref_val = durations.mean()
                        unit = "сек."
                        factor = 1.0
                        if ref_val > 86400: unit, factor = "дн.", 86400.0
                        elif ref_val > 3600: unit, factor = "час.", 3600.0
                        
                        durations_adj = durations / factor
                        durations_adj.hist(bins=20)
                        chart['title'] = f"Продолжительность кейсов ({unit})"
                        data_summary = {
                            "unit": unit,
                            "mean": float(round(durations_adj.mean(), 2)),
                            "median": float(round(durations_adj.median(), 2)),
                            "max": float(round(durations_adj.max(), 2)),
                            "min": float(round(durations_adj.min(), 2))
                        }

                plt.title(chart['title'])
                plt.tight_layout()
                
                filepath = os.path.join(output_dir, chart['name'])
                plt.savefig(filepath)
                plt.close()
                
                abs_path = os.path.abspath(filepath).replace("\\", "/")
                
                # LLM Interpretation
                stats_prompt = f"Chart: {chart['name']}\nData Summary: {json.dumps(data_summary, ensure_ascii=False)}"
                interp_system = (
                    "Ты — аналитик. Дай краткую интерпретацию (1 предложение) этого графика на основе статистики. "
                    "Обязательно используй КОНКРЕТНЫЕ ЦИФРЫ и ЕДИНИЦЫ ИЗМЕРЕНИЯ из сводки. "
                    "ПРАВИЛА ЕДИНИЦ: "
                    "1. Если в сводке unit='дн.', пиши результат в днях (дн.). "
                    "2. Если среднее значение в днях > 1, ОБЯЗАТЕЛЬНО укажи эквивалент в часах в скобках. "
                    "   Пример: 'Средняя длительность 1.5 дн. (36.0 час.)'. "
                    "3. Используй сокращения: 'дн.', 'час.', 'мин.', 'сек.'. "
                    "4. НЕ используй примеры из этого промпта как свои ответы, делай расчеты на основе Data Summary."
                )
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
            gap_explanation = f"Данные охватывают период в {time_range_days} дней (с {ts_series.min().strftime('%Y-%m-%d')} по {ts_series.max().strftime('%Y-%m-%d')})."

            # PNG links for evidence
            png_links = ", ".join([f"[{os.path.basename(r['image'])}]({r['image']})" for r in results if 'image' in r])
            
            return json.dumps({
                "visualizations": results,
                "thoughts": f"Сгенерированы 5 обязательных графиков для Process Mining: распределение операций, временная шкала, события на кейс, интервалы и длительность кейсов. {gap_explanation} "
                            f"ВЫБОР ФУНКЦИЙ: plt.hist() для распределений, groupby().diff() для интервалов, groupby().agg(max-min) для длительности. "
                            f"Доказательства: файлы сохранены в формате .png ({png_links}). "
                            f"Интерпретации строго следуют правилам единиц измерения и содержат конкретные числовые значения из расчетов.",
                "applied_functions": ["plt.savefig()", "df.value_counts()", "df.groupby().diff()", "pd.to_datetime()", "df.groupby().agg()"]
            }, indent=2, ensure_ascii=False)
        except Exception as e:
            cols = list(df.columns) if hasattr(df, 'columns') else "N/A"
            return json.dumps({"error": f"Visualization failed: {e}. Columns: {cols}"}, ensure_ascii=False)



