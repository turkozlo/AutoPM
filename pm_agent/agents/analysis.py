import pandas as pd
import pm4py
import json
import numpy as np
import os
from typing import Dict, Any, Tuple

class ProcessAnalysisAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def get_best_unit(self, seconds: float) -> Tuple[float, str]:
        """Returns (scaled_value, unit_name)"""
        if seconds < 60:
            return seconds, "сек."
        elif seconds < 3600:
            return seconds / 60, "мин."
        elif seconds < 86400:
            return seconds / 3600, "час."
        else:
            return seconds / 86400, "дн."

    def run(self, pm_columns: Dict[str, str], output_dir: str = ".", feedback: str = "") -> str:
        """
        Analyzes process performance and returns strict analysis_result.json.
        """
        df = self.df.copy() # Isolate
        self.df_orig = df.copy() # For debugging
        
        # 0. Deduplicate column names (common issue with dirty CSVs)
        if df.columns.duplicated().any():
            new_cols = []
            counts = {}
            for col in df.columns:
                if col in counts:
                    counts[col] += 1
                    new_cols.append(f"{col}_{counts[col]}")
                else:
                    counts[col] = 0
                    new_cols.append(col)
            df.columns = new_cols

        # 0.5 Drop existing pm4py columns IMMEDIATELY to avoid any conflicts
        pm4py_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
        df = df.drop(columns=[c for c in pm4py_cols if c in df.columns])
        # Normalize keys and handle variations
        def get_col(keys, d):
            for k in keys:
                for dk, dv in d.items():
                    dk_norm = dk.lower().replace(" ", "").replace("_", "").replace(":", "")
                    k_norm = k.lower().replace(" ", "").replace("_", "").replace(":", "")
                    if dk_norm == k_norm:
                        # Ensure we return a single string if dv is a list
                        return dv[0] if isinstance(dv, list) else str(dv)
            return None

        case_col = get_col(['case_id', 'caseid', 'case', 'case:concept:name'], pm_columns)
        activity_col = get_col(['activity', 'event', 'operation', 'concept:name'], pm_columns)
        timestamp_col = get_col(['timestamp', 'time', 'date', 'time:timestamp'], pm_columns)

        # Validation and Fallback (Same as DiscoveryAgent)
        if not activity_col or activity_col not in df.columns:
            for c in df.columns:
                if 'activity' in c.lower() or 'operation' in c.lower() or 'event' in c.lower():
                    activity_col = c
                    break
        
        if not timestamp_col or timestamp_col not in df.columns:
             for c in df.columns:
                if 'time' in c.lower() or 'date' in c.lower():
                    timestamp_col = c
                    break

        if not case_col or (case_col not in df.columns and case_col != 'case_id_synth'):
             for c in df.columns:
                if 'case' in c.lower() or 'id' in c.lower():
                    case_col = c
                    break

        # 1.2 Ensure Timestamp is Datetime (CRITICAL for pm4py)
        if timestamp_col in df.columns:
            try:
                # Handle duplicate column names (df[col] returns a DataFrame)
                ts_data = df[timestamp_col]
                if isinstance(ts_data, pd.DataFrame):
                    ts_data = ts_data.iloc[:, 0]
                
                # FORCE conversion to datetime64[ns]
                df[timestamp_col] = pd.to_datetime(ts_data, errors='coerce')
                
                # Drop rows where timestamp couldn't be parsed
                df = df.dropna(subset=[timestamp_col])
                
                # CRITICAL: Convert to pydatetime for pm4py compatibility
                # We use a list comprehension to ensure we get native python datetime objects
                df[timestamp_col] = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in df[timestamp_col]]
            except Exception as e:
                col_type = str(type(df[timestamp_col]))
                return json.dumps({"error": f"Failed to convert timestamp column '{timestamp_col}' (Type: {col_type}) to datetime: {e}"}, ensure_ascii=False)

        # 1. Synthetic Case ID if needed
        use_synthetic = False
        if not case_col or (case_col not in df.columns and case_col != 'case_id_synth'):
            use_synthetic = True
        elif case_col in df.columns and df[case_col].nunique() > len(df) * 0.9:
            use_synthetic = True
            
        if use_synthetic:
            if timestamp_col in df.columns:
                df = df.sort_values(timestamp_col)
                # Re-ensure datetime for synth calculation
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df['case_id_synth'] = (df[timestamp_col].diff() > pd.Timedelta("30min")).cumsum()
                case_col = 'case_id_synth'
                # Re-convert to pydatetime after synth calculation
                df[timestamp_col] = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in df[timestamp_col]]
            else:
                # Last resort: just index
                df['case_id_synth'] = df.index // 10
                case_col = 'case_id_synth'

        # 1.7 Drop existing pm4py columns to avoid conflicts
        pm4py_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
        cols_to_drop = [c for c in pm4py_cols if c in df.columns and c not in [case_col, activity_col, timestamp_col]]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # 2. Format DataFrame
        try:
            formatted_df = pm4py.format_dataframe(
                df,
                case_id=case_col,
                activity_key=activity_col,
                timestamp_key=timestamp_col
            )

            if formatted_df.empty:
                return json.dumps({"error": "pm4py.format_dataframe returned an empty result. Check your column mappings and data types."}, ensure_ascii=False)
        except Exception as e:
             return json.dumps({"error": f"pm4py formatting failed: {e}. Columns: {list(df.columns)}, Activity: {activity_col}"}, ensure_ascii=False)

        try:
            # 3. Performance Metrics (Python Fact)
            case_durations_sec = pm4py.get_all_case_durations(formatted_df)
            
            # Use pm4py standard names after formatting
            case_col_std = 'case:concept:name'
            activity_col_std = 'concept:name'
            timestamp_col_std = 'time:timestamp'

            duration_stats = {}
            if case_durations_sec is not None and len(case_durations_sec) > 0:
                val_mean, unit_mean = self.get_best_unit(np.mean(case_durations_sec))
                val_median, unit_median = self.get_best_unit(np.median(case_durations_sec))
                val_p95, unit_p95 = self.get_best_unit(np.percentile(case_durations_sec, 95))
                val_max, unit_max = self.get_best_unit(np.max(case_durations_sec))
                
                duration_stats = {
                    "mean": {"value": round(val_mean, 2), "unit": unit_mean},
                    "median": {"value": round(val_median, 2), "unit": unit_median},
                    "p95": {"value": round(val_p95, 2), "unit": unit_p95},
                    "max": {"value": round(val_max, 2), "unit": unit_max}
                }
            
            # Bottlenecks
            sorted_df = formatted_df.sort_values([case_col_std, timestamp_col_std])
            sorted_df['duration_to_next'] = sorted_df.groupby(case_col_std)[timestamp_col_std].diff().shift(-1).dt.total_seconds()
            
            bottlenecks = []
            if 'duration_to_next' in sorted_df.columns:
                b_stats = sorted_df.groupby(activity_col_std)['duration_to_next'].mean().sort_values(ascending=False).head(5)
                for act, dur in b_stats.items():
                    if not pd.isna(dur):
                        val, unit = self.get_best_unit(dur)
                        bottlenecks.append({
                            "activity": act,
                            "mean_duration": round(val, 2),
                            "unit": unit
                        })

            # Performance DFG and PNG
            abs_perf_dfg_path = None
            try:
                perf_dfg, start_acts, end_acts = pm4py.discover_performance_dfg(formatted_df)
                perf_dfg_path = os.path.join(output_dir, "process_performance_dfg.png")
                pm4py.save_vis_performance_dfg(perf_dfg, start_acts, end_acts, perf_dfg_path)
                abs_perf_dfg_path = os.path.abspath(perf_dfg_path).replace("\\", "/")
            except Exception as vis_e:
                print(f"Warning: Performance DFG visualization failed (likely missing Graphviz): {vis_e}")
                # Fallback: Bar chart of bottlenecks
                try:
                    import matplotlib.pyplot as plt
                    if bottlenecks:
                        plt.figure(figsize=(10, 6))
                        acts = [b['activity'] for b in bottlenecks]
                        durs = [b['mean_duration'] for b in bottlenecks]
                        unit = bottlenecks[0]['unit']
                        plt.barh(acts[::-1], durs[::-1], color='salmon')
                        plt.title(f"Top Bottlenecks (Mean Duration in {unit})")
                        plt.xlabel(f"Duration ({unit})")
                        plt.tight_layout()
                        
                        fallback_path = os.path.join(output_dir, "process_performance_bottlenecks.png")
                        plt.savefig(fallback_path)
                        plt.close()
                        abs_perf_dfg_path = os.path.abspath(fallback_path).replace("\\", "/")
                except Exception as plt_e:
                    print(f"Fallback visualization failed: {plt_e}")

            # Loops (discovered from DFG)
            dfg_basic, _, _ = pm4py.discover_dfg(formatted_df)
            loops = [{"activity": k[0], "count": int(v)} for k, v in dfg_basic.items() if k[0] == k[1]]

            # Anomalies (Long cases)
            anomalies = []
            if case_durations_sec is not None and len(case_durations_sec) > 0:
                threshold = np.percentile(case_durations_sec, 99)
                case_groups = formatted_df.groupby(case_col_std)[timestamp_col_std]
                case_durs = (case_groups.max() - case_groups.min()).dt.total_seconds()
                long_cases = case_durs[case_durs >= threshold].head(5)
                for cid, dur in long_cases.items():
                    val, unit = self.get_best_unit(dur)
                    anomalies.append({
                        "case_id": str(cid),
                        "duration": round(val, 2),
                        "unit": unit
                    })

            # Detailed Performance Summary for the Judge
            time_range_days = (formatted_df[timestamp_col_std].max() - formatted_df[timestamp_col_std].min()).days
            mean_str = f"{duration_stats['mean']['value']} {duration_stats['mean']['unit']}" if duration_stats else "N/A"
            p95_str = f"{duration_stats['p95']['value']} {duration_stats['p95']['unit']}" if duration_stats else "N/A"
            median_str = f"{duration_stats['median']['value']} {duration_stats['median']['unit']}" if duration_stats else "N/A"
            
            perf_summary = (
                f"АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ: Среднее время выполнения кейса составляет {mean_str}, медиана {median_str}, при этом 95% кейсов завершаются в пределах {p95_str}. "
                f"УЗКИЕ МЕСТА (Bottlenecks): Наибольшие задержки наблюдаются в операциях: {', '.join([f'{b['activity']} ({b['mean_duration']} {b['unit']})' for b in bottlenecks[:3]]) if bottlenecks else 'не обнаружены'}. "
                f"ЦИКЛЫ: Обнаружено {len(loops)} повторных операций, что может указывать на неэффективность. "
                f"ПЕРИОД: Анализ охватывает {time_range_days} дн. "
                f"ДОКАЗАТЕЛЬСТВА: Расчеты выполнены через pm4py.get_all_case_durations(). {f'Визуализация производительности сохранена в [process_performance_dfg.png]({abs_perf_dfg_path}).' if abs_perf_dfg_path else ''}"
            )

            result = {
                "cases": int(formatted_df[case_col_std].nunique()),
                "case_duration": duration_stats,
                "bottlenecks": bottlenecks,
                "loops": loops,
                "anomalies": anomalies,
                "image_performance": abs_perf_dfg_path,
                "thoughts": perf_summary,
                "applied_functions": ["pm4py.get_all_case_durations()", "pm4py.discover_performance_dfg()", "pm4py.save_vis_performance_dfg()", "df.groupby().diff()"]
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)



        except Exception as e:
            return json.dumps({"error": f"Process analysis failed: {e}"}, ensure_ascii=False)


