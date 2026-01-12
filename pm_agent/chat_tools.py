"""
Chat Tools for Interactive QA Mode.
Comprehensive library of pandas-based analysis functions.
These functions can be called by the agent during chat to perform dynamic analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


# ============================================
# BASIC INFO FUNCTIONS
# ============================================

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Returns basic info about the DataFrame: shape, columns, dtypes."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }


def get_column_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Returns detailed statistics for a specific column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    col = df[column]
    result = {
        "column": column,
        "dtype": str(col.dtype),
        "non_null_count": int(col.count()),
        "null_count": int(col.isna().sum()),
        "unique_count": int(col.nunique())
    }
    
    if pd.api.types.is_numeric_dtype(col):
        result.update({
            "mean": round(float(col.mean()), 4) if not col.isna().all() else None,
            "median": round(float(col.median()), 4) if not col.isna().all() else None,
            "std": round(float(col.std()), 4) if not col.isna().all() else None,
            "min": float(col.min()) if not col.isna().all() else None,
            "max": float(col.max()) if not col.isna().all() else None,
            "sum": float(col.sum()) if not col.isna().all() else None
        })
    
    return result


def get_describe(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
    """Returns pandas describe() output for specified or all numeric columns."""
    try:
        if columns:
            desc = df[columns].describe()
        else:
            desc = df.describe()
        return desc.to_dict()
    except Exception as e:
        return {"error": str(e)}


# ============================================
# VALUE COUNTS & UNIQUE
# ============================================

def get_value_counts(df: pd.DataFrame, column: str, top_n: int = 10, normalize: bool = False) -> Dict[str, Any]:
    """Returns value counts for a column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    vc = df[column].value_counts(normalize=normalize).head(top_n)
    return {
        "column": column,
        "total_unique": int(df[column].nunique()),
        "values": [{"value": str(idx), "count": float(v) if normalize else int(v)} for idx, v in vc.items()]
    }


def get_unique_values(df: pd.DataFrame, column: str, limit: int = 50) -> Dict[str, Any]:
    """Returns unique values in a column (up to limit)."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    unique = df[column].dropna().unique()
    return {
        "column": column,
        "total_unique": len(unique),
        "sample_values": [str(v) for v in unique[:limit]]
    }


# ============================================
# FILTERING
# ============================================

def filter_by_value(df: pd.DataFrame, column: str, value: Any) -> Dict[str, Any]:
    """Filters DataFrame by exact value match, returns count and sample."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    filtered = df[df[column] == value]
    return {
        "original_rows": len(df),
        "filtered_rows": len(filtered),
        "percentage": round(len(filtered) / len(df) * 100, 2) if len(df) > 0 else 0,
        "sample": filtered.head(5).to_dict(orient='records') if len(filtered) > 0 else []
    }


def filter_numeric_range(df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None) -> Dict[str, Any]:
    """Filters DataFrame by numeric range."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    filtered = df.copy()
    if min_val is not None:
        filtered = filtered[filtered[column] >= min_val]
    if max_val is not None:
        filtered = filtered[filtered[column] <= max_val]
    
    return {
        "original_rows": len(df),
        "filtered_rows": len(filtered),
        "percentage": round(len(filtered) / len(df) * 100, 2) if len(df) > 0 else 0
    }


def filter_contains(df: pd.DataFrame, column: str, substring: str, case_sensitive: bool = False) -> Dict[str, Any]:
    """Filters DataFrame where column contains substring."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    filtered = df[df[column].astype(str).str.contains(substring, case=case_sensitive, na=False)]
    return {
        "original_rows": len(df),
        "filtered_rows": len(filtered),
        "percentage": round(len(filtered) / len(df) * 100, 2) if len(df) > 0 else 0
    }


# ============================================
# AGGREGATION & GROUPING
# ============================================

def group_and_count(df: pd.DataFrame, group_by: str, top_n: int = 10) -> Dict[str, Any]:
    """Groups by column and counts occurrences."""
    if group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found"}
    
    grouped = df.groupby(group_by).size().sort_values(ascending=False).head(top_n)
    return {
        "group_by": group_by,
        "total_groups": int(df[group_by].nunique()),
        "top_groups": [{"group": str(idx), "count": int(v)} for idx, v in grouped.items()]
    }


def group_and_aggregate(df: pd.DataFrame, group_by: str, agg_column: str, agg_func: str = "mean", top_n: int = 10) -> Dict[str, Any]:
    """Groups by column and aggregates another column (mean, sum, count, min, max, median)."""
    if group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found"}
    if agg_column not in df.columns:
        return {"error": f"Column '{agg_column}' not found"}
    
    valid_funcs = ["mean", "sum", "count", "min", "max", "median", "std"]
    if agg_func not in valid_funcs:
        return {"error": f"Invalid agg_func. Use one of: {valid_funcs}"}
    
    grouped = df.groupby(group_by)[agg_column].agg(agg_func).sort_values(ascending=False).head(top_n)
    return {
        "group_by": group_by,
        "agg_column": agg_column,
        "agg_func": agg_func,
        "results": [{"group": str(idx), "value": round(float(v), 4) if pd.notna(v) else None} for idx, v in grouped.items()]
    }


# ============================================
# SORTING & TOP/BOTTOM N
# ============================================

def get_top_n(df: pd.DataFrame, column: str, n: int = 10, ascending: bool = False) -> Dict[str, Any]:
    """Returns top N rows sorted by column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    sorted_df = df.sort_values(by=column, ascending=ascending).head(n)
    return {
        "sorted_by": column,
        "ascending": ascending,
        "rows": sorted_df.to_dict(orient='records')
    }


# ============================================
# CORRELATION & RELATIONSHIPS
# ============================================

def get_correlation(df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
    """Calculates correlation between two numeric columns."""
    if column1 not in df.columns or column2 not in df.columns:
        return {"error": "One or both columns not found"}
    
    try:
        corr = df[column1].corr(df[column2])
        return {
            "column1": column1,
            "column2": column2,
            "correlation": round(float(corr), 4) if pd.notna(corr) else None
        }
    except Exception as e:
        return {"error": str(e)}


def get_correlation_matrix(df: pd.DataFrame, columns: List[str] = None) -> Dict[str, Any]:
    """Returns correlation matrix for numeric columns."""
    try:
        if columns:
            corr = df[columns].corr()
        else:
            corr = df.select_dtypes(include=[np.number]).corr()
        return corr.to_dict()
    except Exception as e:
        return {"error": str(e)}


# ============================================
# PERCENTILES & QUANTILES
# ============================================

def get_percentile(df: pd.DataFrame, column: str, percentile: float) -> Dict[str, Any]:
    """Returns the value at a given percentile (0-100)."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    try:
        value = df[column].quantile(percentile / 100)
        return {
            "column": column,
            "percentile": percentile,
            "value": round(float(value), 4) if pd.notna(value) else None
        }
    except Exception as e:
        return {"error": str(e)}


def get_quantiles(df: pd.DataFrame, column: str, q: List[float] = [0.25, 0.5, 0.75]) -> Dict[str, Any]:
    """Returns multiple quantiles for a column."""
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}
    
    try:
        quantiles = df[column].quantile(q)
        return {
            "column": column,
            "quantiles": {f"{int(k*100)}%": round(float(v), 4) for k, v in quantiles.items()}
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================
# PROCESS MINING SPECIFIC
# ============================================

def calculate_path_frequency(df: pd.DataFrame, case_col: str, activity_col: str, top_n: int = 5) -> Dict[str, Any]:
    """Calculates the frequency of unique process paths (traces)."""
    try:
        if 'timestamp' in df.columns:
            traces = df.sort_values(by=[case_col, 'timestamp']).groupby(case_col)[activity_col].apply(tuple)
        else:
            traces = df.groupby(case_col)[activity_col].apply(tuple)
        
        total_cases = len(traces)
        path_counts = traces.value_counts()
        top_paths = path_counts.head(top_n)
        
        result = {
            "total_unique_paths": len(path_counts),
            "total_cases": total_cases,
            "top_paths": []
        }
        
        for path, count in top_paths.items():
            percentage = (count / total_cases) * 100
            result["top_paths"].append({
                "path": " -> ".join(path[:5]) + ("..." if len(path) > 5 else ""),
                "full_length": len(path),
                "count": int(count),
                "percentage": round(percentage, 2)
            })
        
        return result
    except Exception as e:
        return {"error": str(e)}


def get_rarest_paths(df: pd.DataFrame, case_col: str, activity_col: str, bottom_n: int = 5) -> Dict[str, Any]:
    """Finds the RAREST (least frequent) process paths."""
    try:
        if 'timestamp' in df.columns:
            traces = df.sort_values(by=[case_col, 'timestamp']).groupby(case_col)[activity_col].apply(tuple)
        else:
            traces = df.groupby(case_col)[activity_col].apply(tuple)
        
        total_cases = len(traces)
        path_counts = traces.value_counts()
        
        # Get bottom N (rarest)
        rarest_paths = path_counts.tail(bottom_n).sort_values(ascending=True)
        
        result = {
            "total_unique_paths": len(path_counts),
            "total_cases": total_cases,
            "rarest_paths": []
        }
        
        for path, count in rarest_paths.items():
            percentage = (count / total_cases) * 100
            result["rarest_paths"].append({
                "path": " -> ".join(path[:5]) + ("..." if len(path) > 5 else ""),
                "full_length": len(path),
                "count": int(count),
                "percentage": round(percentage, 4)  # More precision for rare paths
            })
        
        return result
    except Exception as e:
        return {"error": str(e)}


def get_case_duration_stats(df: pd.DataFrame, case_col: str, timestamp_col: str) -> Dict[str, Any]:
    """Calculates duration statistics for cases."""
    try:
        df_temp = df.copy()
        df_temp[timestamp_col] = pd.to_datetime(df_temp[timestamp_col])
        
        case_durations = df_temp.groupby(case_col)[timestamp_col].agg(['min', 'max'])
        case_durations['duration_seconds'] = (case_durations['max'] - case_durations['min']).dt.total_seconds()
        
        return {
            "total_cases": len(case_durations),
            "mean_duration_hours": round(case_durations['duration_seconds'].mean() / 3600, 2),
            "median_duration_hours": round(case_durations['duration_seconds'].median() / 3600, 2),
            "min_duration_hours": round(case_durations['duration_seconds'].min() / 3600, 2),
            "max_duration_hours": round(case_durations['duration_seconds'].max() / 3600, 2),
            "std_duration_hours": round(case_durations['duration_seconds'].std() / 3600, 2)
        }
    except Exception as e:
        return {"error": str(e)}


def get_activity_frequency(df: pd.DataFrame, activity_col: str, top_n: int = 10) -> Dict[str, Any]:
    """Returns frequency of activities."""
    try:
        freq = df[activity_col].value_counts().head(top_n)
        total = len(df)
        
        return {
            "total_events": total,
            "unique_activities": int(df[activity_col].nunique()),
            "top_activities": [
                {"activity": str(act), "count": int(cnt), "percentage": round((cnt/total)*100, 2)}
                for act, cnt in freq.items()
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def count_cases_and_activities(df: pd.DataFrame, case_col: str, activity_col: str) -> Dict[str, Any]:
    """Returns basic counts."""
    try:
        return {
            "total_events": len(df),
            "unique_cases": int(df[case_col].nunique()),
            "unique_activities": int(df[activity_col].nunique()),
            "avg_events_per_case": round(len(df) / df[case_col].nunique(), 2)
        }
    except Exception as e:
        return {"error": str(e)}


def find_bottlenecks(df: pd.DataFrame, case_col: str, activity_col: str, timestamp_col: str, top_n: int = 5) -> Dict[str, Any]:
    """Finds activities with longest average waiting time before them."""
    try:
        df_temp = df.copy()
        df_temp[timestamp_col] = pd.to_datetime(df_temp[timestamp_col])
        df_temp = df_temp.sort_values(by=[case_col, timestamp_col])
        
        df_temp['prev_timestamp'] = df_temp.groupby(case_col)[timestamp_col].shift(1)
        df_temp['wait_time'] = (df_temp[timestamp_col] - df_temp['prev_timestamp']).dt.total_seconds() / 3600  # hours
        
        bottlenecks = df_temp.groupby(activity_col)['wait_time'].mean().sort_values(ascending=False).head(top_n)
        
        return {
            "bottlenecks": [
                {"activity": str(act), "avg_wait_hours": round(float(wait), 2)}
                for act, wait in bottlenecks.items() if pd.notna(wait)
            ]
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================
# TOOL REGISTRY
# ============================================

CHAT_TOOLS = {
    # Basic Info
    "get_dataframe_info": {
        "function": get_dataframe_info,
        "description": "Получить базовую информацию о DataFrame: размер, колонки, типы данных.",
        "required_args": []
    },
    "get_column_stats": {
        "function": get_column_stats,
        "description": "Получить статистику по конкретной колонке (среднее, медиана, мин, макс и др).",
        "required_args": ["column"]
    },
    "get_describe": {
        "function": get_describe,
        "description": "Получить describe() статистику для числовых колонок.",
        "required_args": [],
        "optional_args": ["columns"]
    },
    
    # Value Counts
    "get_value_counts": {
        "function": get_value_counts,
        "description": "Подсчитать количество каждого уникального значения в колонке.",
        "required_args": ["column"],
        "optional_args": ["top_n", "normalize"]
    },
    "get_unique_values": {
        "function": get_unique_values,
        "description": "Получить список уникальных значений в колонке.",
        "required_args": ["column"],
        "optional_args": ["limit"]
    },
    
    # Filtering
    "filter_by_value": {
        "function": filter_by_value,
        "description": "Отфильтровать данные по точному значению в колонке.",
        "required_args": ["column", "value"]
    },
    "filter_numeric_range": {
        "function": filter_numeric_range,
        "description": "Отфильтровать данные по диапазону числовых значений.",
        "required_args": ["column"],
        "optional_args": ["min_val", "max_val"]
    },
    "filter_contains": {
        "function": filter_contains,
        "description": "Отфильтровать данные где колонка содержит подстроку.",
        "required_args": ["column", "substring"],
        "optional_args": ["case_sensitive"]
    },
    
    # Aggregation
    "group_and_count": {
        "function": group_and_count,
        "description": "Сгруппировать по колонке и посчитать количество в каждой группе.",
        "required_args": ["group_by"],
        "optional_args": ["top_n"]
    },
    "group_and_aggregate": {
        "function": group_and_aggregate,
        "description": "Сгруппировать и агрегировать (mean, sum, count, min, max, median, std).",
        "required_args": ["group_by", "agg_column", "agg_func"],
        "optional_args": ["top_n"]
    },
    
    # Sorting
    "get_top_n": {
        "function": get_top_n,
        "description": "Получить топ N строк отсортированных по колонке.",
        "required_args": ["column"],
        "optional_args": ["n", "ascending"]
    },
    
    # Correlation
    "get_correlation": {
        "function": get_correlation,
        "description": "Вычислить корреляцию между двумя числовыми колонками.",
        "required_args": ["column1", "column2"]
    },
    "get_correlation_matrix": {
        "function": get_correlation_matrix,
        "description": "Получить матрицу корреляций для числовых колонок.",
        "required_args": [],
        "optional_args": ["columns"]
    },
    
    # Percentiles
    "get_percentile": {
        "function": get_percentile,
        "description": "Получить значение на заданном перцентиле (0-100).",
        "required_args": ["column", "percentile"]
    },
    "get_quantiles": {
        "function": get_quantiles,
        "description": "Получить квантили (по умолчанию 25%, 50%, 75%).",
        "required_args": ["column"],
        "optional_args": ["q"]
    },
    
    # Process Mining
    "calculate_path_frequency": {
        "function": calculate_path_frequency,
        "description": "Получить ТОП N самых ЧАСТЫХ путей (трейсов). Используй для 'покажи топ путей', 'самый частый путь'.",
        "required_args": ["case_col", "activity_col"],
        "optional_args": ["top_n"]
    },
    "get_case_duration_stats": {
        "function": get_case_duration_stats,
        "description": "Рассчитать статистику длительности кейсов (среднее, медиана и др).",
        "required_args": ["case_col", "timestamp_col"]
    },
    "get_activity_frequency": {
        "function": get_activity_frequency,
        "description": "Получить частоту активностей (событий).",
        "required_args": ["activity_col"],
        "optional_args": ["top_n"]
    },
    "count_cases_and_activities": {
        "function": count_cases_and_activities,
        "description": "Получить базовые подсчеты: события, кейсы, активности.",
        "required_args": ["case_col", "activity_col"]
    },
    "find_bottlenecks": {
        "function": find_bottlenecks,
        "description": "Найти узкие места (активности с наибольшим временем ожидания).",
        "required_args": ["case_col", "activity_col", "timestamp_col"],
        "optional_args": ["top_n"]
    },
    "get_rarest_paths": {
        "function": get_rarest_paths,
        "description": "Найти САМЫЕ РЕДКИЕ пути (наименее частые трейсы процесса).",
        "required_args": ["case_col", "activity_col"],
        "optional_args": ["bottom_n"]
    }
}


def get_tools_description() -> str:
    """Returns a formatted description of available tools for the LLM."""
    lines = ["Доступные инструменты анализа:"]
    for name, info in CHAT_TOOLS.items():
        args = ", ".join(info.get("required_args", []))
        lines.append(f"- **{name}**({args}): {info['description']}")
    return "\n".join(lines)


def execute_tool(tool_name: str, args: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Executes a tool by name with given arguments."""
    if tool_name not in CHAT_TOOLS:
        return {"error": f"Неизвестный инструмент: {tool_name}. Попробуй один из: {', '.join(CHAT_TOOLS.keys())}"}
    
    tool_info = CHAT_TOOLS[tool_name]
    func = tool_info["function"]
    
    try:
        return func(df, **args)
    except TypeError as e:
        return {"error": f"Неверные аргументы: {e}"}
    except Exception as e:
        return {"error": str(e)}
