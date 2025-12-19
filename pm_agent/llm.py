from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from .config import MISTRAL_MODEL, MISTRAL_API_KEY

class LLMClient:
    def __init__(self):
        self.client = ChatMistralAI(
            model=MISTRAL_MODEL,
            api_key=MISTRAL_API_KEY,
            temperature=0.2
        )

    def generate_response(self, prompt: str, system_prompt: str = "Ты полезный ИИ-ассистент.") -> str:
        """
        Generates a response from the LLM with retry logic for rate limits.
        """
        import time
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.invoke(messages)
                return response.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Превышен лимит запросов (429). Ожидание {wait_time} сек...")
                    time.sleep(wait_time)
                    continue
                print(f"Ошибка при генерации ответа: {e}")
                return f"Ошибка: {e}"
        return "Ошибка: Превышено количество попыток."

    def analyze_data_cleaning(self, data_head: str, data_info: str, feedback: str = "") -> str:
        """
        Asks LLM how to clean the data based on head and info.
        Returns a JSON-like string or instructions.
        """
        system_prompt = (
            "Ты опытный Data Scientist. Твоя задача — проанализировать структуру датасета и предложить операции по очистке. "
            "Ты получишь первые несколько строк (head) и вывод info(). "
            "Определи столбцы с пропущенными значениями (NaN) и реши, что делать: "
            "1. Удалить строку (drop_row), если отсутствуют критические данные. "
            "2. Заполнить средним (fill_mean) для числовых данных. "
            "3. Заполнить модой (fill_mode) или заполнителем для категориальных данных. "
            "Верни ответ в виде валидного JSON списка действий, например: "
            '[{"column": "Age", "action": "fill_mean"}, {"column": "ID", "action": "drop_row"}]'
        )
        prompt = f"Data Head:\n{data_head}\n\nData Info:\n{data_info}"
        if feedback:
            prompt += f"\n\nЗАМЕЧАНИЯ СУДЬИ (ИСПРАВЬ ЭТО): {feedback}"
        
        return self.generate_response(prompt, system_prompt)

    def judge_step(self, step_name: str, context: str, result: str) -> dict:
        """
        Evaluates the result of a step.
        Returns a dict with 'passed' (bool), 'critique' (str), and 'score' (int).
        """
        system_prompt = (
            "Ты строгий и придирчивый Судья (Judge). Твоя задача — оценивать качество выполнения шага агентом Process Mining. "
            "Ты должен быть БЕСКОМПРОМИССНЫМ. Если результат не идеален — ОТКЛОНЯЙ его. "
            "ПРАВИЛА: "
            "1. Результат должен быть полным, точным и соответствовать задаче. "
            "2. Используй ТОЛЬКО тот чек-лист, который соответствует названию шага (step_name). "
            "3. НИКАКИХ упоминаний технических ограничений. "
            "4. ВАЖНО: Валидные пути к файлам (.png) ЯВЛЯЮТСЯ ДОКАЗАТЕЛЬСТВОМ. "
            "5. Требуй конкретики и цифр. Общие фразы недопустимы. "
            "ЧЕК-ЛИСТЫ: "
            "- Data Profiling: есть ли nan_percent, unique, top_10 для колонок? Есть ли оценка готовности к PM? "
            "- Data Cleaning: указано ли сколько строк удалено/заполнено? Соответствует ли факт плану? (Использование mean/mode для заполнения — это НОРМАЛЬНО). ВАЖНО: Для этого шага НЕ ТРЕБУЮТСЯ файлы .png. "
            "- Visualization: есть ли ссылки на 4 обязательных файла (operation_distribution, timestamp_distribution, case_duration, inter_event_time)? Есть ли цифры в интерпретации? Используются ли удобные единицы (мин/час/дни)? ВАЖНО: Если данные охватывают годы, то 'дн.' (дни) — это АДЕКВАТНАЯ единица. Проверяй поле 'thoughts' на наличие объяснений и ссылок на доказательства. "
            "- Process Discovery: есть ли количество активностей/переходов? Есть ли блок Mermaid? "
            "- Process Analysis: есть ли среднее время и p95? Указаны ли конкретные узкие места? Используются ли адекватные единицы (сек/мин/час/дни)? "
            "ВАЖНО: Если данные охватывают годы, то 'дн.' (дни) — это АДЕКВАТНАЯ единица. Проверяй поле 'thoughts' на наличие объяснений и ссылок на доказательства. "
            "КРИТЕРИЙ АДЕКВАТНОСТИ: Если процесс длится минуты, а агент пишет '0 дней' или '1000000 мс' — ОТКЛОНЯЙ. "
            "Верни JSON с полями: "
            "- 'passed': true/false "
            "- 'critique': жесткая критика (если false) "
            "- 'score': оценка от 1 до 10"
        )

        prompt = f"Шаг: {step_name}\nКонтекст: {context}\nРезультат агента: {result}"
        response = self.generate_response(prompt, system_prompt)
        
        try:
            import json
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
            return {"passed": True, "critique": "Не удалось разобрать ответ Судьи, пропускаем.", "score": 5}
        except:
            return {"passed": True, "critique": "Ошибка парсинга ответа Судьи.", "score": 5}

    def reflect_on_result(self, context: str, result: str) -> dict:
        # Legacy reflection method, kept for compatibility but Judge is preferred now
        return self.judge_step("Reflection", context, result)
