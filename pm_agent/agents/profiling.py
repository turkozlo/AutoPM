import json


import pandas as pd


class DataProfilingAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def run(self, feedback: str = "") -> str:
        """
        Calculates strict metrics and uses LLM to interpret them.
        Accepts 'feedback' from Judge to refine analysis.
        """
        total_rows = len(self.df)
        col_stats = {}

        for col in self.df.columns:
            col_data = self.df[col]
            nan_count = int(col_data.isna().sum())
            unique_count = int(col_data.nunique())

            stats = {
                "dtype": str(col_data.dtype),
                "nan": nan_count,
                "nan_percent": float(round((nan_count / total_rows) * 100, 6)),
                "unique": unique_count,
            }

            # Mode
            if not col_data.dropna().empty:
                stats["mode"] = str(col_data.mode()[0])

            # Top 10 values (useful for LLM to identify column semantic)
            top_counts = col_data.value_counts().head(10)
            stats["top_10"] = [
                {
                    "value": str(k)[:100],
                    "count": int(v),
                    "percent": float(round((v / total_rows) * 100, 6)),
                }
                for k, v in top_counts.items()
            ]

            col_stats[col] = stats

        # Ask LLM to analyze
        system_prompt = (
            "Ты — Data Profiling Agent. Твоя задача — проанализировать статистику колонок датасета для задачи Process Mining. "
            "ПРАВИЛА: "
            "1. Изучи 'col_stats'. Определи ЛУЧШИЕ кандидаты на роли: Case ID, Activity, Timestamp. "
            "2. Рассчитай скор готовности (0-100%). "
            "   - Case ID обязателен (обычно уникальный или почти уникальный для кейса). "
            "   - Activity обязателен (категориальный, не слишком много уникальных). "
            "   - Timestamp обязателен (формат даты/времени). Если это object, укажи в рекомендациях преобразовать. "
            "3. Если есть 'feedback' от Судьи, УЧТИ ЕГО. Например, если Судья сказал, что Case ID выбран неверно, выбери другой. "
            "4. Верни JSON с полями: 'process_mining_readiness' (score, level, reasons, recommendations, "
            "case_id_candidates, activity_candidates, timestamp_candidates, justifications) и 'thoughts'. "
            "В 'thoughts' объясни свой выбор."
        )

        prompt = f"Column Stats:\n{json.dumps(col_stats, indent=2, ensure_ascii=False)}"
        if feedback:
            prompt += f"\n\nКРИТИКА СУДЬИ (ИСПРАВЬ ОШИБКИ): {feedback}"

        response = self.llm.generate_response(prompt, system_prompt)

        try:
            # Parse LLM response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                analysis = json.loads(response[start:end])
            else:
                raise ValueError("No JSON found")

            # Merge with raw stats
            result = {
                "row_count": total_rows,
                "column_count": len(self.df.columns),
                "columns": col_stats,
                "duplicates": int(self.df.duplicated().sum()),
                "process_mining_readiness": analysis.get(
                    "process_mining_readiness", {}
                ),
                "thoughts": analysis.get("thoughts", "Analysis completed."),
                "applied_functions": ["df.describe", "df.nunique", "llm.analyze"],
            }
            return json.dumps(result, indent=2, ensure_ascii=False)

        except Exception as e:
            return json.dumps(
                {"error": f"LLM Analysis failed: {e}. Raw: {response}"},
                ensure_ascii=False,
            )
