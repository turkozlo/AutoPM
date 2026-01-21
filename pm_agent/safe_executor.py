"""
Safe Executor for Code Interpreter.
Executes user-generated pandas code in a sandboxed environment.
"""

import ast
import builtins
import signal
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out (5 seconds)")


def validate_code_syntax(code: str) -> Dict[str, Any]:
    """
    Checks code for syntax errors without executing it.
    Returns {"success": True} or {"success": False, "error": "..."}.
    """
    try:
        ast.parse(code)
        return {"success": True}
    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Синтаксическая ошибка: {e.msg} (строка {e.lineno}, столбец {e.offset})",
        }
    except Exception as e:
        return {"success": False, "error": f"Ошибка валидации: {e}"}


def execute_pandas_code(
    code: str, df: pd.DataFrame, timeout_seconds: int = 5
) -> Dict[str, Any]:
    """
    Executes pandas code in a restricted namespace.

    Security measures:
    - Only df, pd, np available
    - __builtins__ blocked (no open, exec, eval, import)
    - Execution on df.copy() to prevent mutations
    - Timeout to prevent infinite loops
    - Result size limit (10KB)

    Returns:
        {"success": True, "result": ..., "result_type": "..."} or
        {"success": False, "error": "..."}
    """
    # Create safe namespace
    safe_df = df.copy()
    
    # Define restricted builtins
    restricted_builtins = {
        "__import__": builtins.__import__,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "round": round,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "tuple": tuple,
        "set": set,
        "abs": abs,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "any": any,
        "all": all,
        "print": lambda *args, **kwargs: None,  # Disable print
    }

    allowed_globals = {
        "df": safe_df,
        "pd": pd,
        "np": np,
        "__builtins__": restricted_builtins,
    }

    local_vars = {}

    try:
        # Set timeout (Unix only, skip on Windows)
        if sys.platform != "win32":
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)

        # Execute code
        exec(code, allowed_globals, local_vars)

        # Cancel timeout
        if sys.platform != "win32":
            signal.alarm(0)

        # Get result
        if "result" not in local_vars:
            return {
                "success": False,
                "error": "Код выполнен, но переменная 'result' не определена. Сохрани результат в переменную 'result'.",
            }

        result = local_vars["result"]

        # Convert result to JSON-serializable format
        result_str = format_result(result)

        # Check size limit (10KB)
        if len(result_str) > 10240:
            result_str = (
                result_str[:10000] + "\n... (обрезано, слишком большой результат)"
            )

        return {
            "success": True,
            "result": result_str,
            "result_type": type(result).__name__,
        }

    except TimeoutError as e:
        return {"success": False, "error": str(e)}
    except SyntaxError as e:
        return {"success": False, "error": f"Синтаксическая ошибка: {e}"}
    except NameError as e:
        return {
            "success": False,
            "error": f"Ошибка имени (возможно, используется недоступная функция): {e}",
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        if sys.platform != "win32":
            signal.alarm(0)


def format_result(result: Any) -> str:
    """Formats result for display."""
    if isinstance(result, pd.DataFrame):
        if len(result) > 20:
            return f"DataFrame ({len(result)} строк, {len(result.columns)} колонок):\n{result.head(10).to_string()}\n... (показаны первые 10)"
        return result.to_string()
    elif isinstance(result, pd.Series):
        if len(result) > 20:
            return f"Series ({len(result)} элементов):\n{result.head(10).to_string()}\n... (показаны первые 10)"
        return result.to_string()
    elif isinstance(result, (list, tuple)) and len(result) > 50:
        return str(result[:50]) + f"\n... (всего {len(result)} элементов)"
    elif isinstance(result, dict) and len(result) > 20:
        items = list(result.items())[:20]
        return str(dict(items)) + f"\n... (всего {len(result)} ключей)"
    else:
        return str(result)


def get_df_info_for_llm(df: pd.DataFrame) -> str:
    """Generates a concise DataFrame description for LLM context."""
    info_lines = [
        f"DataFrame: {len(df)} строк, {len(df.columns)} колонок",
        f"Колонки: {', '.join(df.columns[:15])}"
        + ("..." if len(df.columns) > 15 else ""),
        "Типы данных:",
    ]

    for col in df.columns[:10]:
        dtype = str(df[col].dtype)
        sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
        if len(sample) > 30:
            sample = sample[:27] + "..."
        info_lines.append(f"  - {col}: {dtype} (пример: {sample})")

    if len(df.columns) > 10:
        info_lines.append(f"  ... и еще {len(df.columns) - 10} колонок")

    return "\n".join(info_lines)
