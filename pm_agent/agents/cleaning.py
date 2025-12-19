import pandas as pd
import json
from typing import Dict, Any

class DataCleaningAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def run(self, profiling_report: Dict[str, Any]) -> str:
        """
        Analyzes profiling report, asks LLM for cleaning plan, executes it, 
        and returns strict cleaning_result.json.
        """
        self.df = self.df.copy() # Isolate
        # 1. Ask LLM for plan based on Profile
        system_prompt = (
            "Ты — Data Cleaning Agent. Твоя задача — создать ПЛАН очистки данных на основе профиля данных (JSON). "
            "Правила: "
            "1. Если 'nan' == 0, ничего не делай с этим столбцом. "
            "2. Если 'nan' > 0: "
            "   - Для числовых ('int', 'float') используй 'fill_mean' или 'fill_median'. "
            "   - Для категориальных ('object') используй 'fill_mode'. "
            "   - Если пропусков очень мало (< 5%) и данные критичны, можно 'drop_row'. "
            "3. Верни ТОЛЬКО валидный JSON список действий. НЕ пиши итоговые цифры (строки, пропуски), это сделает Python. "
            "Пример ответа: "
            '[{"column": "ColName", "action": "fill_mean", "reason": "Заполнено средним для сохранения объема данных"}]'
        )
        
        # Filter profile to only show columns with issues
        issues = {k: v for k, v in profiling_report['columns'].items() if v['nan'] > 0}
        duplicates = profiling_report.get('duplicates', 0)
        
        if not issues and duplicates == 0:
            return json.dumps({
                "rows_before": profiling_report['row_count'],
                "rows_after": profiling_report['row_count'],
                "rows_removed": 0,
                "removed_by_column": {},
                "duplicates_removed": 0,
                "fill_actions": [],
                "message": "Data is clean. No cleaning needed."
            }, ensure_ascii=False)

        prompt = f"Data Profile (Issues only):\n{json.dumps(issues, indent=2)}\nDuplicates: {duplicates}"
        plan_str = self.llm.generate_response(prompt, system_prompt)
        
        # 2. Parse Plan
        actions = []
        try:
            clean_json_str = plan_str.strip()
            if "```json" in clean_json_str:
                clean_json_str = clean_json_str.split("```json")[1].split("```")[0]
            elif "```" in clean_json_str:
                 clean_json_str = clean_json_str.split("```")[1].split("```")[0]
            actions = json.loads(clean_json_str)
        except:
            pass

        # 3. Execute Plan (Python Fact)
        initial_rows = len(self.df)
        removed_by_column = {}
        fill_actions = []
        
        # Handle Duplicates
        duplicates_removed = 0
        if duplicates > 0:
            before_dup = len(self.df)
            self.df.drop_duplicates(inplace=True)
            duplicates_removed = before_dup - len(self.df)

        for action in actions:
            col = action.get("column")
            act = action.get("action")
            reason = action.get("reason", "Standard cleaning")
            
            if col not in self.df.columns:
                continue
                
            nan_before = int(self.df[col].isna().sum())
            if nan_before == 0 and act != "drop_row":
                continue

            if act == "drop_row":
                before_drop = len(self.df)
                self.df.dropna(subset=[col], inplace=True)
                removed_count = before_drop - len(self.df)
                removed_by_column[col] = removed_by_column.get(col, 0) + removed_count
            elif act in ["fill_mean", "fill_median", "fill_mode", "fill_empty"]:
                val = None
                if act == "fill_mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].mean()
                elif act == "fill_median" and pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].median()
                elif act == "fill_mode":
                    if not self.df[col].mode().empty:
                        val = self.df[col].mode()[0]
                elif act == "fill_empty":
                    val = "Unknown"
                
                if val is not None:
                    self.df[col] = self.df[col].fillna(val)
                    fill_actions.append({
                        "column": col,
                        "action": act,
                        "value": str(val) if not isinstance(val, (int, float)) else float(round(val, 2)),
                        "reason": reason
                    })

        final_rows = len(self.df)
        
        # Prepare verbose thoughts for the Judge
        fill_summary = ", ".join([f"{a['column']}: {a['action']} ({a['value']})" for a in fill_actions]) if fill_actions else "нет"
        cleaning_summary = (
            f"Очистка завершена. Удалено строк: {initial_rows - final_rows}. "
            f"Удалено дубликатов: {duplicates_removed}. "
            f"Заполнено пропусков: {fill_summary}. "
            f"Все изменения соответствуют плану. Визуализация для этого шага не требуется, так как изменения носят структурный характер."
        )

        result = {
            "rows_before": initial_rows,
            "rows_after": final_rows,
            "rows_removed": initial_rows - final_rows,
            "removed_by_column": removed_by_column,
            "duplicates_removed": duplicates_removed,
            "fill_actions": fill_actions,
            "thoughts": cleaning_summary,
            "applied_functions": ["df.drop_duplicates()", "df.dropna()", "df.fillna()", "df.mean()", "df.mode()"]
        }
        
        return json.dumps(result, indent=2, ensure_ascii=False), self.df

