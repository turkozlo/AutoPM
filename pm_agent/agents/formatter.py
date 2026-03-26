import pandas as pd
import numpy as np
import json
import re

class DataFormatterAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def _is_datetime_like(self, series: pd.Series) -> bool:
        """Heuristic to detect if a column looks like it contains dates."""
        # 1. Если уже datetime — отлично
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # 2. Если это объекты (строки или Timestamp-объекты)
        if series.dtype == 'object':
            sample = series.dropna().head(20).astype(str)
            if sample.empty:
                return False
            
            # Регулярки для ISO дат (2025-01-01, 2025-01-01T12:00:00, etc)
            # И чуть более общие паттерны
            date_patterns = [
                r'^\d{4}-\d{2}-\d{2}',           # 2025-01-01
                r'^\d{2}\.\d{2}\.\d{4}',         # 01.01.2025
                r'^\d{4}/\d{2}/\d{2}',           # 2025/01/01
                r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}' # 2025-01-01T12:00
            ]
            
            matches = 0
            for val in sample:
                val = val.strip()
                if any(re.search(p, val) for p in date_patterns):
                    matches += 1
            
            # Если хотя бы 30% непустых строк совпало — считаем датой
            return (matches / len(sample)) > 0.3
            
        return False

    def run(self) -> pd.DataFrame:
        """
        Analyzes column types and converts them (int, float, datetime).
        Aggressively forces dates to a consistent format.
        """
        print("Starting Data Formatting...")
        df_new = self.df.copy()
        
        # 1. Сначала пытаемся угадать типы через LLM (для бизнес-логики)
        type_map = {}
        try:
            sample_str = self.df.head(5).to_string()
            if len(sample_str) > 2000:
                sample_str = sample_str[:2000] + "\n... (truncated)"
            
            system_prompt = (
                "Identify the correct data type for each column.\n"
                "Supported: 'int', 'float', 'datetime', 'string'.\n"
                "Return JSON: {\"col_name\": \"type\"}"
            )
            prompt = f"Data sample:\n{sample_str}\n\nColumns: {list(self.df.columns)}"
            response = self.llm.generate_response(prompt, system_prompt)
            type_map = self.llm._parse_json(response) or {}
        except Exception as e:
            print(f"LLM Type Detection warning: {e}")

        # 2. Применяем конвертации
        for col in df_new.columns:
            target_type = type_map.get(col)
            
            # АГРЕССИВНАЯ ПРОВЕРКА НА ДАТЫ (даже если LLM ошибся)
            if target_type == 'datetime' or self._is_datetime_like(df_new[col]):
                try:
                    # ТОТАЛЬНАЯ ЗАЧИСТКА ДАТ:
                    # 1. Конвертируем в UTC (обрабатывает и строки, и Timestamp-объекты)
                    # 2. Убираем таймзону (делаем наивной), чтобы numpy/pm4py не ругались
                    # 3. Принудительно ставим тип datetime64[ns]
                    df_new[col] = pd.to_datetime(
                        df_new[col], utc=True, errors='coerce'
                    ).dt.tz_localize(None).astype('datetime64[ns]')
                    print(f"Column '{col}' FORCED to datetime64[ns]")
                    continue # Переходим к следующей колонке
                except Exception as e:
                    print(f"Error forced-converting '{col}' to datetime: {e}")

            # Числовые типы
            if target_type in ['int', 'float']:
                try:
                    df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    if target_type == 'int' and df_new[col].notna().all():
                        df_new[col] = df_new[col].astype('int64')
                    print(f"Column '{col}' converted to {target_type}")
                except Exception as e:
                    print(f"Error converting '{col}' to {target_type}: {e}")

        return df_new
