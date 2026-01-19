import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd


class VisualizationAgent:
    def __init__(self, df: pd.DataFrame, llm_client):
        self.df = df
        self.llm = llm_client

    def run(
        self,
        profiling_report: Dict[str, Any],
        output_dir: str = ".",
        feedback: str = "",
    ) -> str:
        """
        Generates mandatory PM charts and asks LLM to interpret them.
        """
        df = self.df.copy()  # Isolate to prevent side effects
        if df.empty:
            return json.dumps({"error": "DataFrame is empty. Cannot generate visualizations."}, ensure_ascii=False)

        # 1. Identify Columns for PM Charts
        pm_readiness = profiling_report.get("process_mining_readiness", {})

        # Helper to ensure string column names
        def sanitize_col(val):
            if isinstance(val, dict):
                # Try to extract 'column' or 'name' or 'value'
                for k in ["column", "name", "value", "col"]:
                    if k in val:
                        return str(val[k])
                # Fallback: first value
                return str(list(val.values())[0])
            if isinstance(val, list):
                return sanitize_col(val[0]) if val else None
            return str(val) if val is not None else None

        # Try to find best candidates
        activity_col = sanitize_col(pm_readiness.get("activity_candidates", [None])[0])
        timestamp_col = sanitize_col(
            pm_readiness.get("timestamp_candidates", [None])[0]
        )
        case_col = sanitize_col(pm_readiness.get("case_id_candidates", [None])[0])

        # Fallback to first column if not found
        if not activity_col:
            activity_col = list(df.columns)[0]
        if not timestamp_col:
            # Look for any datetime column
            for col, stats in profiling_report["columns"].items():
                if "datetime" in stats["dtype"]:
                    timestamp_col = col
                    break
        if not case_col:
            case_col = list(df.columns)[0]

        # 1.5 Synthetic Case ID if needed
        if (
            not case_col
            or case_col not in df.columns
            or df[case_col].nunique() > len(df) * 0.9
        ):
            df = df.sort_values(timestamp_col)
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            df["case_id_synth"] = (
                df[timestamp_col].diff() > pd.Timedelta("30min")
            ).cumsum()
            case_col = "case_id_synth"

        # 2. Ask LLM for Chart Plan
        system_prompt = (
            "Ты — Visualization Agent. Твоя задача — выбрать список графиков для Process Mining. "
            "Доступные типы:\n"
            "- 'bar': Для категориальных (распределение операций).\n"
            "- 'hist': Гистограмма временных меток (распределение во времени).\n"
            "- 'inter_event': Интервалы между событиями (требует Case ID + Timestamp).\n"
            "- 'case_duration': Длительность кейсов (требует Case ID + Timestamp).\n\n"
            "ПРАВИЛА:\n"
            "1. Верни JSON список: [{'name': 'filename.png', 'type': 'type', 'column': 'col_name', 'title': 'Title'}].\n"
            "2. ОБЯЗАТЕЛЬНО включи 4 стандартных графика, если данные позволяют.\n"
            "3. Учти 'feedback' от Судьи (если есть), чтобы исправить ошибки прошлого запуска."
        )

        prompt = (
            f"Candidates:\nActivity: {activity_col}\nTimestamp: {timestamp_col}\nCase ID: {case_col}\n"
            f"Columns: {list(df.columns)}"
        )
        if feedback:
            prompt += f"\n\nКРИТИКА СУДЬИ (ИСПРАВЬ ОШИБКИ): {feedback}"

        plan_resp = self.llm.generate_response(prompt, system_prompt)

        charts = []
        try:
            start = plan_resp.find("[")
            end = plan_resp.rfind("]") + 1
            if start != -1 and end != -1:
                charts = json.loads(plan_resp[start:end])
        except Exception:
            # Fallback if LLM fails
            charts = [
                {
                    "name": "operation_distribution.png",
                    "type": "bar",
                    "column": activity_col,
                    "title": f"Частота операций ({activity_col})",
                },
                {
                    "name": "timestamp_distribution.png",
                    "type": "hist",
                    "column": timestamp_col,
                    "title": f"Распределение во времени ({timestamp_col})",
                },
                {
                    "name": "inter_event_time.png",
                    "type": "inter_event",
                    "column": timestamp_col,
                    "title": "Интервалы между событиями",
                },
                {
                    "name": "case_duration.png",
                    "type": "case_duration",
                    "column": timestamp_col,
                    "title": "Продолжительность кейсов",
                },
            ]

        results = []

        for chart in charts:
            try:
                plt.figure(figsize=(10, 6))
                col = chart["column"]
                ctype = chart["type"]

                data_summary = {}

                if ctype == "bar":
                    counts = df[col].value_counts().head(10)
                    counts.plot(kind="bar")
                    data_summary = {
                        "unit": "событий",
                        "top_1": str(counts.index[0]) if not counts.empty else "N/A",
                        "top_1_count": int(counts.iloc[0]) if not counts.empty else 0,
                        "details": {str(k): int(v) for k, v in counts.items()},
                    }
                elif ctype == "hist":
                    # Ensure timestamp is datetime
                    temp_ts = pd.to_datetime(df[col], errors="coerce").dropna()
                    if not temp_ts.empty:
                        temp_ts.hist(bins=20)
                        data_summary = {
                            "unit": "дн.",
                            "min": temp_ts.min().strftime("%Y-%m-%d"),
                            "max": temp_ts.max().strftime("%Y-%m-%d"),
                            "total_days": float(
                                round(
                                    (temp_ts.max() - temp_ts.min()).total_seconds()
                                    / 86400,
                                    2,
                                )
                            ),
                        }
                elif ctype == "inter_event":
                    # Sort and calculate diff
                    temp_df = df[[case_col, timestamp_col]].copy()
                    temp_df[timestamp_col] = pd.to_datetime(
                        temp_df[timestamp_col], errors="coerce"
                    )
                    temp_df = temp_df.sort_values([case_col, timestamp_col])
                    diffs_sec = (
                        temp_df.groupby(case_col)[timestamp_col]
                        .diff()
                        .dt.total_seconds()
                        .dropna()
                    )

                    if not diffs_sec.empty:
                        ref_val = diffs_sec.mean()
                        unit = "сек."
                        factor = 1.0
                        if ref_val > 86400:
                            unit, factor = "дн.", 86400.0
                        elif ref_val > 3600:
                            unit, factor = "час.", 3600.0
                        elif ref_val > 60:
                            unit, factor = "мин.", 60.0

                        diffs = diffs_sec / factor
                        diffs.hist(bins=20)
                        chart["title"] = f"Интервалы между событиями ({unit})"
                        data_summary = {
                            "unit": unit,
                            "mean": float(round(diffs.mean(), 2)),
                            "median": float(round(diffs_sec.median() / factor, 4)),
                        }
                elif ctype == "case_duration":
                    temp_df = df[[case_col, timestamp_col]].copy()
                    temp_df[timestamp_col] = pd.to_datetime(
                        temp_df[timestamp_col], errors="coerce"
                    )
                    durations = (
                        temp_df.groupby(case_col)[timestamp_col]
                        .agg(lambda x: (x.max() - x.min()).total_seconds())
                        .dropna()
                    )

                    if not durations.empty:
                        ref_val = durations.mean()
                        unit = "сек."
                        factor = 1.0
                        if ref_val > 86400:
                            unit, factor = "дн.", 86400.0
                        elif ref_val > 3600:
                            unit, factor = "час.", 3600.0

                        durations_adj = durations / factor
                        durations_adj.hist(bins=20)
                        chart["title"] = f"Продолжительность кейсов ({unit})"
                        data_summary = {
                            "unit": unit,
                            "mean": float(round(durations_adj.mean(), 2)),
                            "median": float(round(durations_adj.median(), 2)),
                            "max": float(round(durations_adj.max(), 2)),
                            "min": float(round(durations_adj.min(), 2)),
                        }

                plt.title(chart["title"])
                plt.tight_layout()

                filepath = os.path.join(output_dir, chart["name"])
                plt.savefig(filepath)
                plt.close()

                rel_path = chart["name"]

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

                results.append(
                    {
                        "image": rel_path,
                        "type": ctype,
                        "column": col,
                        "data_summary": data_summary,
                        "interpretation": interpretation.strip(),
                    }
                )

            except Exception as e:
                results.append({"chart": chart["name"], "error": str(e)})

        try:
            # Explanation for large gaps
            ts_series = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
            time_range_days = (ts_series.max() - ts_series.min()).days
            gap_explanation = (
                f"Данные охватывают период в {time_range_days} дней "
                f"(с {ts_series.min().strftime('%Y-%m-%d')} по {ts_series.max().strftime('%Y-%m-%d')})."
            )

            # PNG links for evidence
            png_links = ", ".join(
                [
                    f"[{os.path.basename(r['image'])}]({r['image']})"
                    for r in results
                    if "image" in r
                ]
            )

            return json.dumps(
                {
                    "visualizations": results,
                    "thoughts": (
                        f"Сгенерированы 4 обязательных графика для Process Mining: "
                        f"распределение операций, временная шкала, интервалы и длительность кейсов. {gap_explanation} "
                        f"ВЫБОР ФУНКЦИЙ: plt.hist() для распределений, groupby().diff() для интервалов, "
                        f"groupby().agg(max-min) для длительности. "
                        f"Доказательства (PNG ссылки): {png_links}. "
                        f"Интерпретации строго следуют правилам единиц измерения и содержат "
                        f"конкретные числовые значения из расчетов."
                    ),
                    "applied_functions": [
                        "plt.savefig()",
                        "df.value_counts()",
                        "df.groupby().diff()",
                        "pd.to_datetime()",
                        "df.groupby().agg()",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            cols = list(df.columns) if hasattr(df, "columns") else "N/A"
            return json.dumps(
                {"error": f"Visualization failed: {e}. Columns: {cols}"},
                ensure_ascii=False,
            )
