import pandas as pd
import pm4py
import json
import os
from typing import Dict, Any

class ProcessDiscoveryAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def run(self, pm_columns: Dict[str, str] = None, output_dir: str = ".") -> str:
        """
        Discovers process model and returns strict discovery_result.json.
        """
        df = self.df.copy() # Isolate
        self.df_orig = df.copy() # For debugging
        
        # 0. Deduplicate column names
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

        # 1. Identify Columns (if needed)
        if not pm_columns:
            system_prompt = (
                "Ты — Process Mining Expert. Твоя задача — найти столбцы Case ID, Activity и Timestamp. "
                "Верни ТОЛЬКО JSON: "
                '{"case_id": "ColName", "activity": "ColName", "timestamp": "ColName"}'
            )
            prompt = f"Columns: {df.columns.tolist()}\nHead:\n{df.head(3).to_string()}"
            
            resp = self.llm.generate_response(prompt, system_prompt)
            try:
                json_str = resp.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                     json_str = json_str.split("```")[1].split("```")[0]
                pm_columns = json.loads(json_str)
            except:
                return json.dumps({"error": "Failed to identify PM columns"}, ensure_ascii=False)

        # Normalize keys and handle variations
        def get_col(keys, d):
            for k in keys:
                # Try exact, then lowercase, then stripped lowercase with underscores/spaces removed
                for dk, dv in d.items():
                    dk_norm = dk.lower().replace(" ", "").replace("_", "").replace(":", "")
                    k_norm = k.lower().replace(" ", "").replace("_", "").replace(":", "")
                    if dk_norm == k_norm:
                        # Ensure we return a single string if dv is a list
                        return dv[0] if isinstance(dv, list) else str(dv)
            return None

        case_id = get_col(['case_id', 'caseid', 'case', 'case:concept:name'], pm_columns)
        activity_key = get_col(['activity', 'event', 'operation', 'concept:name'], pm_columns)
        timestamp_key = get_col(['timestamp', 'time', 'date', 'time:timestamp'], pm_columns)
        
        # Validation and Fallback
        if not activity_key or activity_key not in df.columns:
            # Try to find something that looks like activity
            for c in df.columns:
                if 'activity' in c.lower() or 'operation' in c.lower() or 'event' in c.lower():
                    activity_key = c
                    break
        
        if not timestamp_key or timestamp_key not in df.columns:
             for c in df.columns:
                if 'time' in c.lower() or 'date' in c.lower():
                    timestamp_key = c
                    break

        if not case_id or (case_id not in df.columns and case_id != 'case_id_synth'):
             for c in df.columns:
                if 'case' in c.lower() or 'id' in c.lower():
                    case_id = c
                    break

        # Standardize for other agents
        pm_columns = {
            'case_id': case_id,
            'activity': activity_key,
            'timestamp': timestamp_key
        }

        # 1.2 Ensure Timestamp is Datetime (CRITICAL for pm4py)
        if timestamp_key in df.columns:
            try:
                # Handle duplicate column names
                ts_data = df[timestamp_key]
                if isinstance(ts_data, pd.DataFrame):
                    ts_data = ts_data.iloc[:, 0]

                # FORCE conversion to datetime64[ns]
                # Use a more robust approach: if it's already datetime-like, pd.to_datetime is fine.
                # If it's object, try to convert to datetime, and if that fails, try string conversion first.
                df[timestamp_key] = pd.to_datetime(ts_data, errors='coerce')
                
                # Drop rows where timestamp couldn't be parsed
                df = df.dropna(subset=[timestamp_key])
                
                # CRITICAL: Convert to pydatetime for pm4py compatibility
                # We use a list comprehension to ensure we get native python datetime objects
                df[timestamp_key] = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in df[timestamp_key]]
            except Exception as e:
                col_type = str(type(df[timestamp_key]))
                return json.dumps({"error": f"Failed to convert timestamp column '{timestamp_key}' (Type: {col_type}) to datetime: {e}"}, ensure_ascii=False)
        
        # 1.5 Synthetic Case ID if needed
        if not case_id or df[case_id].nunique() > len(df) * 0.9:
            df = df.sort_values(timestamp_key)
            # Re-ensure datetime for synth calculation
            df[timestamp_key] = pd.to_datetime(df[timestamp_key])
            df['case_id_synth'] = (df[timestamp_key].diff() > pd.Timedelta("30min")).cumsum()
            case_id = 'case_id_synth'
            pm_columns['case_id'] = case_id
            # Re-convert to pydatetime after synth calculation
            df[timestamp_key] = [t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t for t in df[timestamp_key]]
        
        # 1.7 Drop existing pm4py columns to avoid conflicts
        pm4py_cols = ['case:concept:name', 'concept:name', 'time:timestamp']
        cols_to_drop = [c for c in pm4py_cols if c in df.columns and c not in [case_id, activity_key, timestamp_key]]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # 2. Format DataFrame
        try:
            if df.empty:
                cols_info = {c: str(self.df_orig[c].dtype) if c in self.df_orig.columns else "N/A" for c in [case_id, activity_key, timestamp_key]}
                return json.dumps({
                    "error": "DataFrame is empty after processing. Cannot perform discovery.",
                    "debug_info": {
                        "initial_rows": len(self.df_orig),
                        "identified_columns": {
                            "case": case_id,
                            "activity": activity_key,
                            "timestamp": timestamp_key
                        },
                        "column_types": cols_info,
                        "available_columns": list(self.df_orig.columns)
                    }
                }, ensure_ascii=False)
                
            formatted_df = pm4py.format_dataframe(
                df,
                case_id=case_id,
                activity_key=activity_key,
                timestamp_key=timestamp_key
            )
            
            if formatted_df.empty:
                return json.dumps({"error": "pm4py.format_dataframe returned an empty result. Check your column mappings and data types."}, ensure_ascii=False)
                
        except Exception as e:
             return json.dumps({"error": f"pm4py formatting failed: {e}"}, ensure_ascii=False)

        try:
            # 3. Discovery Stats (Python Fact)
            start_activities = pm4py.get_start_activities(formatted_df)
            end_activities = pm4py.get_end_activities(formatted_df)
            dfg, start_acts, end_acts = pm4py.discover_dfg(formatted_df)
            
            # Save DFG as PNG for evidence
            abs_dfg_path = None
            try:
                dfg_path = os.path.join(output_dir, "process_discovery_dfg.png")
                pm4py.save_vis_dfg(dfg, start_acts, end_acts, dfg_path)
                abs_dfg_path = os.path.abspath(dfg_path).replace("\\", "/")
            except Exception as vis_e:
                print(f"Warning: DFG visualization failed (likely missing Graphviz): {vis_e}")
                # Fallback: Bar chart of top transitions
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    # We need transitions list here, let's calculate it first
                    temp_transitions = []
                    for (act_from, act_to), count in dfg.items():
                        temp_transitions.append({"from": act_from, "to": act_to, "count": int(count)})
                    temp_transitions.sort(key=lambda x: x['count'], reverse=True)
                    
                    top_t = temp_transitions[:10]
                    labels = [f"{t['from']} -> {t['to']}" for t in top_t]
                    counts = [t['count'] for t in top_t]
                    plt.barh(labels[::-1], counts[::-1], color='skyblue')
                    plt.title("Top 10 Transitions (Frequency)")
                    plt.xlabel("Count")
                    plt.tight_layout()
                    
                    fallback_path = os.path.join(output_dir, "process_discovery_top_transitions.png")
                    plt.savefig(fallback_path)
                    plt.close()
                    abs_dfg_path = os.path.abspath(fallback_path).replace("\\", "/")
                except Exception as plt_e:
                    print(f"Fallback visualization failed: {plt_e}")
            
            transitions = []
            total_transitions_count = 0
            for (act_from, act_to), count in dfg.items():
                c = int(count)
                transitions.append({
                    "from": act_from,
                    "to": act_to,
                    "count": c
                })
                total_transitions_count += c
            transitions.sort(key=lambda x: x['count'], reverse=True)

            # Loops
            loops = [{"activity": t['from'], "count": t['count']} for t in transitions if t['from'] == t['to']]

            # Mermaid Diagram Generation (Robust version with IDs)
            mermaid_code = ""
            if transitions:
                mermaid_lines = ["graph TD"]
                # Add styling
                mermaid_lines.append("    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px,color:#333,font-size:12px;")
                mermaid_lines.append("    classDef startNode fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
                mermaid_lines.append("    classDef endNode fill:#fff3e0,stroke:#e65100,stroke-width:2px;")
                
                node_map = {}
                node_id_counter = 1
                
                # Identify top start/end for styling
                top_starts = sorted(start_activities.items(), key=lambda x: x[1], reverse=True)[:3]
                top_ends = sorted(end_activities.items(), key=lambda x: x[1], reverse=True)[:3]
                starts = [k for k, v in top_starts]
                ends = [k for k, v in top_ends]

                # Use a threshold to keep the diagram clean but connected
                max_count = transitions[0]['count'] if transitions else 0
                threshold = max_count * 0.05 # Show transitions with at least 5% of max count
                
                filtered_transitions = [t for t in transitions if t['count'] >= threshold]
                if len(filtered_transitions) > 50:
                    filtered_transitions = transitions[:50]
                elif len(filtered_transitions) < 15:
                    filtered_transitions = transitions[:15]

                for t in filtered_transitions:
                    # Ensure both nodes have IDs
                    for act in [t['from'], t['to']]:
                        if act not in node_map:
                            n_id = f"node{node_id_counter}"
                            node_map[act] = n_id
                            node_id_counter += 1
                            
                            # Add label and class using ::: syntax
                            label = str(act).replace('"', "'")[:50]
                            style = ""
                            if act in starts:
                                style = ":::startNode"
                            elif act in ends:
                                style = ":::endNode"
                            
                            mermaid_lines.append(f'    {n_id}["{label}"]{style}')
                    
                    f_id = node_map[t['from']]
                    to_id = node_map[t['to']]
                    mermaid_lines.append(f'    {f_id} -->|{t["count"]}| {to_id}')
                
                mermaid_code = "\n".join(mermaid_lines)

            num_activities = int(formatted_df[activity_key].nunique())
            num_edges = len(dfg)
            top_start = max(start_activities.items(), key=lambda x: x[1])[0] if start_activities else "N/A"
            top_end = max(end_activities.items(), key=lambda x: x[1])[0] if end_activities else "N/A"

            result = {
                "pm_columns": pm_columns,
                "activities": num_activities,
                "edges": num_edges,
                "start_activities": [{"activity": k, "count": int(v)} for k, v in start_activities.items()],
                "end_activities": [{"activity": k, "count": int(v)} for k, v in end_activities.items()],
                "top_transitions": transitions[:10],
                "loops": loops,
                "mermaid": mermaid_code,
                "image_dfg": abs_dfg_path,
                "thoughts": f"Процесс успешно восстановлен. ОБЩАЯ СТАТИСТИКА: Обнаружено {num_activities} активностей, {num_edges} уникальных переходов (всего {total_transitions_count} событий перехода) и {len(loops)} циклов (петель). "
                            f"КЛЮЧЕВЫЕ ТОЧКИ: Основная стартовая активность - '{top_start}', основная конечная - '{top_end}'. "
                            f"УЗКИЕ МЕСТА (Bottlenecks): Наиболее частые переходы: {', '.join([f'{t['from']} -> {t['to']} ({t['count']})' for t in transitions[:3]])}. "
                            f"ДОКАЗАТЕЛЬСТВА: Сгенерирована схема Mermaid (см. ниже) и файл [process_discovery_dfg.png]({abs_dfg_path}). Расчеты выполнены через pm4py.discover_dfg().\n\n"
                            f"```mermaid\n{mermaid_code}\n```",
                "applied_functions": ["pm4py.discover_dfg()", "pm4py.save_vis_dfg()", "mermaid_generation"]
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)



        except Exception as e:
            return json.dumps({"error": f"Process discovery failed: {e}"}, ensure_ascii=False)

