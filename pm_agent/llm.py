import time
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from openai import OpenAI
from .config import (
    LOCAL_API_KEY,
    LOCAL_BASE_URL,
    LOCAL_MODEL,
    MISTRAL_API_KEY,
    MISTRAL_MODEL,
    PROVIDER,
    GIGACHAT_BASE_URL,
    GIGACHAT_ACCESS_TOKEN,
    GIGACHAT_MODEL,
)

class LocalLLMClient:
    """Wrapper for native OpenAI client to match langchain interface."""
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.2):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: list, **kwargs) -> "LocalLLMResponse":
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "content"):
                role = "system" if isinstance(msg, SystemMessage) else "user"
                formatted_messages.append({"role": role, "content": msg.content})
            else:
                formatted_messages.append(msg)

        response = self.client.chat.completions.create(
            model=self.model, messages=formatted_messages, temperature=self.temperature, **kwargs
        )
        return LocalLLMResponse(response.choices[0].message.content)

class LocalLLMResponse:
    def __init__(self, content: str):
        self.content = content

class LLMClient:
    def __init__(self, rag_manager=None):
        self.rag_manager = rag_manager
        if PROVIDER == "local":
            self.client = LocalLLMClient(
                base_url=LOCAL_BASE_URL,
                model=LOCAL_MODEL,
                api_key=LOCAL_API_KEY,
                temperature=0.2,
            )
        elif PROVIDER == "gigachat":
            from langchain_gigachat.chat_models import GigaChat
            self.client = GigaChat(
                base_url=GIGACHAT_BASE_URL,
                access_token=GIGACHAT_ACCESS_TOKEN,
                model=GIGACHAT_MODEL,
            )
        else:
            self.client = ChatMistralAI(
                model_name=MISTRAL_MODEL, api_key=MISTRAL_API_KEY, temperature=0.2
            )

    def _parse_json(self, response_str: str) -> dict:
        """Helper for internal needs (formatter.py still uses JSON)."""
        import json
        import re
        if not response_str: return None
        # Clean potential markdown blocks
        clean_str = response_str
        if "```json" in response_str:
            clean_str = response_str.split("```json")[1].split("```")[0].strip()
        elif "```" in response_str:
             clean_str = response_str.split("```")[1].split("```")[0].strip()
        
        try: return json.loads(clean_str)
        except: pass
        
        # Try regex if standard parsing fails
        match = re.search(r"(\{.*\})", clean_str, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        return None

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "Ты полезный ИИ-ассистент.",
        json_mode: bool = False,
    ) -> str:
        """
        Generates a response from the LLM with retry logic for rate limits.
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]

        current_delay = 2.0
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Remove max_tokens argument to let the model generate until it stops naturally
                response = self.client.invoke(messages)
                return response.content
            except Exception as e:
                error_str = str(e)
                if ("429" in error_str or "Too Many Requests" in error_str) and attempt < max_retries - 1:
                    print(f"Превышен лимит запросов (429). Ожидание {current_delay} сек... (Попытка {attempt + 1}/{max_retries})")
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
                    continue
                print(f"Ошибка при генерации ответа: {e}")
                return f"Ошибка: {e}"
        return "Ошибка: Превышено количество попыток."

    def simple_chat(self, user_query: str, context: str, history: list = None) -> dict:
        """
        Smart Router chat: answers directly or signals that code is needed.
        Returns dict: {"answer": str | None, "needs_code": bool}
        """
        import json

        system_prompt = (
            "Ты — Эксперт-консультант по Process Mining с доступом к данным и базе знаний о платформе.\n"
            "ПРАВИЛА:\n"
            "1. Если вопрос касается ФУНКЦИОНАЛА ПЛАТФОРМЫ (как строить графики, как запустить ML, "
            "как удалять данные, интерфейс, настройки) — верни needs_rag: true.\n"
            "2. Если вопрос требует РАСЧЕТОВ по текущему датасету (среднее, медиана, фильтрация, топ-N) "
            "— верни needs_code: true.\n"
            "3. Отвечай на простые вопросы о структуре данных напрямую, если расчет не требуется.\n"
            "4. Формат ответа СТРОГО JSON.\n"
            "\n"
            "ФОРМАТ ОТВЕТА:\n"
            '{"answer": "Текст (если не нужен RAG/код)", "needs_code": false, "needs_rag": false}\n'
            "Если нужен расчет:\n"
            '{"answer": null, "needs_code": true, "needs_rag": false}\n'
            "Если вопрос про инструкцию/платформу:\n"
            '{"answer": null, "needs_code": false, "needs_rag": true}\n'
        )

        # Keyword detection for platform/instructions
        platform_keywords = [
            "как", "инструкция", "руководство", "платформа", "интерфейс", 
            "настройк", "ml модель", "кластеризация", "удалит", "создать"
        ]
        
        # Check for RAG intent first manually as a fallback/accelerator
        if any(kw in user_query.lower() for kw in platform_keywords) and self.rag_manager:
            # Let's see if LLM confirms or just use it
            pass

        prompt_parts = [f"КОНТЕКСТ ДАННЫХ:\n{context}"]
        if history:
            prompt_parts.append("\nИСТОРИЯ ДИАЛОГА:")
            for msg in history[-10:]:
                role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                prompt_parts.append(f"{role}: {msg['text']}")
        prompt_parts.append(f"\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}")
        prompt = "\n".join(prompt_parts)

        response_str = self.generate_response(prompt, system_prompt)
        parsed = self._parse_json(response_str)
        if parsed:
            # RAG Logic
            if parsed.get("needs_rag") and self.rag_manager:
                print("📚 Поиск в базе знаний...")
                docs = self.rag_manager.retrieve(user_query, top_k=2)
                if docs:
                    for d in docs:
                        print(f"   -> [{d['score']:.3f}] {d['title']}")

                    doc_context = "\n\n".join([d["formatted"] for d in docs])
                    source_paths = "\n".join([f"  - {d['path']}" for d in docs])

                    rag_prompt = (
                        f"Используя ТОЛЬКО следующие документы из базы знаний, "
                        f"ответь на вопрос пользователя.\n"
                        f"Если информации недостаточно — честно скажи об этом.\n"
                        f"В конце ответа ОБЯЗАТЕЛЬНО укажи источники.\n\n"
                        f"{doc_context}\n\n"
                        f"ВОПРОС: {user_query}\n\n"
                        f"ИСТОЧНИКИ:\n{source_paths}"
                    )
                    rag_answer = self.generate_response(
                        rag_prompt,
                        "Ты технический писатель Платформы Process Mining. "
                        "Отвечай структурированно и по существу."
                    )
                    return {"answer": rag_answer, "needs_code": False, "needs_rag": True}
                else:
                    return {
                        "answer": "К сожалению, в базе знаний нет информации по вашему вопросу.",
                        "needs_code": False, "needs_rag": True,
                    }

            return parsed

        # Fallback: if JSON parsing failed, return as plain text answer
        return {"answer": response_str, "needs_code": False, "needs_rag": False}

    # ------------------------------------------------------------------
    #  Code Interpreter methods
    # ------------------------------------------------------------------

    def generate_pandas_code(
        self, question: str, df_info: str, previous_error: str = "", context: dict = None
    ) -> dict:
        """
        Generates pandas code to answer the user's question.
        Returns: {"thought": str, "code": str}
        """
        import json

        context = context or {}
        error_context = ""
        if previous_error:
            error_context = f"\n\nПРЕДЫДУЩАЯ ПОПЫТКА ЗАВЕРШИЛАСЬ ОШИБКОЙ:\n{previous_error}\nИСПРАВЬ КОД!\n"

        system_prompt = (
            "Ты — Эксперт по анализу данных. Твоя задача — написать pandas-код для ответа на вопрос пользователя.\n"
            "\n"
            "ДОСТУПНЫЕ ПЕРЕМЕННЫЕ:\n"
            "- df: pandas DataFrame с данными (Process Mining Event Log)\n"
            "- pd: pandas библиотека\n"
            "- np: numpy библиотека\n"
            "- plt: matplotlib.pyplot для визуализации\n"
            "\n"
            "ТЕРМИНОЛОГИЯ PROCESS MINING:\n"
            "- АКТИВНОСТЬ (Activity) = Событие, строка в логе.\n"
            "- ПУТЬ/ТРЕЙС (Trace, Variant) = ПОСЛЕДОВАТЕЛЬНОСТЬ активностей для одного case_id.\n"
            "  Для анализа путей собери их в СТРОКУ: .apply(lambda x: ' -> '.join(x)).\n"
            "\n"
            "ПРАВИЛА:\n"
            "1. ОБЯЗАТЕЛЬНО сохрани финальный результат в переменную 'result'.\n"
            "2. Код должен быть простым и читаемым.\n"
            "3. Используй только pandas/numpy/matplotlib операции.\n"
            "4. ВИЗУАЛИЗАЦИЯ: Если нужен график, сохрани его через `plt.savefig('reports/temp_plot.png')` и установи `result = 'reports/temp_plot.png'`.\n"
            "5. 'result' должен быть стандартным Python типом (int, float, dict, list, str). Используй .item() для numpy скаляров.\n"
            "6. Результат должен быть JSON-сериализуемым.\n"
            "\n"
            "ПРИМЕРЫ (Few-Shot):\n"
            "Вопрос: Сколько всего уникальных кейсов?\n"
            "Код: result = df['case_id'].nunique()\n\n"
            "Вопрос: Найди топ-5 самых частых активностей.\n"
            "Код: result = df['activity'].value_counts().head(5).to_dict()\n\n"
            "Вопрос: Построй график распределения длительности кейсов.\n"
            "Код:\n"
            "durations = df.groupby('case_id')['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)\n"
            "plt.figure(figsize=(10,6))\n"
            "durations.hist(bins=20)\n"
            "plt.title('Распределение длительности кейсов (часы)')\n"
            "plt.savefig('reports/temp_plot.png')\n"
            "result = 'reports/temp_plot.png'\n"
            "\n"
            "7. БЕЗОПАСНАЯ СОРТИРОВКА (LINUX FIX): Если нужно сортировать по колонке с датой, ВСЕГДА используй промежуточный .astype('int64').\n"
            "   Пример: df.assign(ts_int=df['timestamp'].astype('int64')).sort_values(['case_id', 'ts_int']).drop(columns='ts_int')\n"
            "8. НЕ ПЕРЕКОНВЕРТИРУЙ: Даты уже в формате datetime64[ns]. Не вызывай pd.to_datetime() повторно.\n"
            + error_context
            + "\n"
            "ФОРМАТ ОТВЕТА (JSON):\n"
            "{\n"
            '  "thought": "Рассуждение: что нужно сделать и как",\n'
            '  "code": "result = df..."\n'
            "}"
        )

        user_prompt = (
            f"ИНФОРМАЦИЯ О ДАННЫХ:\n{df_info}\n\n"
            f"ВОПРОС:\n{question}"
            f"{error_context}"
        )

        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        parsed = self._parse_json(response_str)
        if parsed and "code" in parsed:
            return parsed

        # Fallback 1: Extract "code" field directly via regex (handles malformed JSON with raw newlines)
        import re
        code_match = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', response_str, re.DOTALL)
        if code_match:
            code = code_match.group(1).replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            # Also try to extract thought
            thought_match = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', response_str, re.DOTALL)
            thought = thought_match.group(1) if thought_match else "Извлечено regex"
            return {"thought": thought, "code": code}

        # Fallback 2: Extract JSON from markdown code fence (```json ... ```)
        for fence_lang in ["```json", "```"]:
            if fence_lang in response_str:
                try:
                    json_block = response_str.split(fence_lang, 1)[1].split("```", 1)[0].strip()
                    inner_parsed = self._parse_json(json_block)
                    if inner_parsed and "code" in inner_parsed:
                        return inner_parsed
                except (IndexError, Exception):
                    pass

        # Fallback 3: Extract python code from markdown block
        if "```python" in response_str:
            try:
                code_block = response_str.split("```python")[1].split("```")[0].strip()
                return {"thought": "Извлечено из markdown блока", "code": code_block}
            except IndexError:
                pass

        # Fallback 4: Direct code assignment
        if "result =" in response_str:
            return {"thought": "Извлечено прямым текстом", "code": response_str.strip()}

        return {"thought": "Не удалось распарсить", "code": response_str}

    def verify_result(self, question: str, result_str: str, history: list = None) -> dict:
        """
        Verifies if the result adequately answers the question considering chat history.
        """
        system_prompt = (
            "Ты — строгий ИИ-Судья. Твоя задача — проверить, является ли результат выполнения кода КАЧЕСТВЕННЫМ ответом на вопрос.\n"
            "\n"
            "КРИТЕРИИ ПРОВЕРКИ:\n"
            "1. ПОЛНОТА: Содержит ли результат все запрашиваемые данные?\n"
            "2. РЕЛЕВАНТНОСТЬ: Соответствует ли ответ смыслу вопроса и контексту диалога?\n"
            "3. ПРАВДОПОДОБНОСТЬ: Не выглядит ли результат заведомо ошибочным (например, пустой список там, где точно должны быть данные)?\n"
            "\n"
            "СПЕЦИАЛЬНЫЕ ПРАВИЛА:\n"
            "- Если результат — путь к файлу (например, 'reports/temp_plot.png'), считай это валидным графиком.\n"
            "- Если пользователь спросил 'Топ 10', а в результате 0 или 1 элемент без объяснения причин — это is_valid: false.\n"
            "\n"
            "ФОРМАТ ОТВЕТА (JSON):\n"
            "{\n"
            '  "is_valid": true | false,\n'
            '  "critique": "Краткое описание проблемы, если есть",\n'
            '  "suggestion": "Как исправить код, чтобы получить верный результат"\n'
            "}\n"
        )

        prompt_parts = []
        if history:
            prompt_parts.append("ИСТОРИЯ ДИАЛОГА:")
            for msg in history[-5:]:
                role = "Пользователь" if msg["role"] == "user" else "Ассистент"
                prompt_parts.append(f"{role}: {msg['text']}")
        
        prompt_parts.append(f"\nВОПРОС: {question}\n\nРЕЗУЛЬТАТ КОДА:\n{result_str}")
        user_prompt = "\n".join(prompt_parts)
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        parsed = self._parse_json(response_str)
        if parsed:
            return parsed

        return {
            "is_valid": False,
            "critique": "Не удалось проверить результат (сбой JSON).",
            "suggestion": "Попробуй выполнить код еще раз.",
        }

    def interpret_code_result(self, question: str, result: str, result_type: str) -> dict:
        """
        Interprets code execution result into a human-friendly answer.
        Returns: {"answer": str}
        """
        system_prompt = (
            "Ты — Эксперт-консультант по Process Mining.\n"
            "Преврати результат выполнения кода в понятный ответ на русском языке.\n"
            "НИКОГДА НЕ ВЫДУМЫВАЙ ЦИФРЫ. Используй ТОЛЬКО факты из результата.\n"
            "Формат ответа: JSON {\"answer\": \"...\"}"
        )

        user_prompt = (
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}\n\n"
            f"РЕЗУЛЬТАТ КОДА (тип: {result_type}):\n{result}"
        )

        if str(result).endswith('.png'):
            user_prompt += "\n\nПРИМЕЧАНИЕ: Результат — это путь к созданному графику. Обязательно упомяни, что график построен и доступен по ссылке."

        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        parsed = self._parse_json(response_str)
        if parsed:
            return parsed

        return {"answer": str(result)}

