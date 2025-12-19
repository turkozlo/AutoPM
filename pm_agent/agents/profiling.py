import pandas as pd
import json
import numpy as np

class DataProfilingAgent:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run(self) -> str:
        """
        Calculates strict metrics and returns a JSON string in the required format.
        """
        total_rows = len(self.df)
        col_stats = {}
        
        numeric_cols = []
        for col in self.df.columns:
            col_data = self.df[col]
            nan_count = int(col_data.isna().sum())
            unique_count = int(col_data.nunique())
            
            stats = {
                "dtype": str(col_data.dtype),
                "nan": nan_count,
                "nan_percent": float(round((nan_count / total_rows) * 100, 2)),
                "unique": unique_count
            }
            
            # Mode
            if not col_data.dropna().empty:
                stats["mode"] = str(col_data.mode()[0])
            
            # Top 10 values
            top_counts = col_data.value_counts().head(10)
            stats["top_10"] = [
                {
                    "value": str(k)[:100],
                    "count": int(v),
                    "percent": float(round((v / total_rows) * 100, 2))
                } for k, v in top_counts.items()
            ]
            
            # Time stats for datetime-like columns
            is_time = False
            if pd.api.types.is_datetime64_any_dtype(col_data):
                is_time = True
            elif col_data.dtype == 'object':
                if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                    try:
                        temp_time = pd.to_datetime(col_data.dropna().head(10), errors='raise')
                        is_time = True
                    except:
                        pass

            if is_time:
                valid_times = pd.to_datetime(col_data, errors='coerce').dropna()
                if not valid_times.empty:
                    stats["min"] = str(valid_times.min())
                    stats["max"] = str(valid_times.max())
                    stats["span_days"] = float(round((valid_times.max() - valid_times.min()).total_seconds() / 86400, 2))

            if pd.api.types.is_numeric_dtype(col_data) and not is_time:
                numeric_cols.append(col)
                stats["min"] = float(col_data.min())
                stats["max"] = float(col_data.max())
                stats["mean"] = float(round(col_data.mean(), 2))
                stats["median"] = float(round(col_data.median(), 2))

            col_stats[col] = stats

        # PM Readiness Candidates
        case_candidates = []
        activity_candidates = []
        timestamp_candidates = []
        
        for col, s in col_stats.items():
            lcol = col.lower()
            if 'id' in lcol or 'case' in lcol or 'number' in lcol:
                case_candidates.append(col)
            if 'activity' in lcol or 'status' in lcol or 'event' in lcol or 'operation' in lcol:
                activity_candidates.append(col)
            if 'date' in lcol or 'time' in lcol or 'timestamp' in lcol:
                timestamp_candidates.append(col)

        profile = {
            "row_count": total_rows,
            "column_count": len(self.df.columns),
            "columns": col_stats,
            "duplicates": int(self.df.duplicated().sum()),
            "process_mining_readiness": {
                "case_id_candidates": case_candidates,
                "activity_candidates": activity_candidates,
                "timestamp_candidates": timestamp_candidates
            },
            "thoughts": "Данные проанализированы. Сформирован детальный профиль с распределением значений и оценкой готовности к Process Mining.",
            "applied_functions": ["df.shape", "df.dtypes", "df.isna().sum()", "df.nunique()", "df.duplicated().sum()", "df.value_counts()"]
        }

        return json.dumps(profile, indent=2, ensure_ascii=False)

