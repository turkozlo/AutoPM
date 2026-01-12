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

    def generate_response(self, prompt: str, system_prompt: str = "Ты полезный ИИ-ассистент.", json_mode: bool = False) -> str:
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
            "Ты строгий, но разумный Судья (Judge). Твоя задача — оценивать качество выполнения шага агентом Process Mining.\n"
            "ПРАВИЛА:\n"
            "1. Результат должен быть полным и полезным. Если данные не идеальны, но агент сделал всё возможное — ПРИНИМАЙ работу.\n"
            "2. Используй чек-лист, соответствующий шагу.\n"
            "3. Если отклоняешь, дай КОНКРЕТНУЮ инструкцию, как исправить. Твоя критика будет передана агенту как обратная связь.\n"
            "4. ВАЖНО: Валидные пути к файлам (.png) — это главное доказательство успеха для визуализаций.\n"
            "5. Будь терпим к мелким недочетам в форматировании, если суть (цифры, графики) верна.\n"
            "ЧЕК-ЛИСТЫ:\n"
            "- Data Profiling: посчитаны ли основные статистики? Есть ли вывод о готовности (даже если он негативный)?\n"
            "- Data Cleaning: есть ли план и отчет о действиях? (Если удалено 0 строк — это ОК, если данные были чистыми).\n"
            "- Visualization: созданы ли графики? (Если какие-то не создались из-за данных — это допустимо, если есть объяснение).\n"
            "- Process Discovery: построена ли схема? (Или объяснено, почему (циклы/шум)).\n"
            "- Process Analysis: есть ли цифры производительности?\n"
            "Верни JSON: {'passed': bool, 'critique': str, 'score': int}"
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

    def judge_session(self, memory: str, final_report: str) -> dict:
        """
        Evaluates the entire session based on Memory and Final Report.
        Returns {'passed': bool, 'critique': str, 'suggested_start_point': str}
        """
        system_prompt = (
            "Ты — Главный Судья (Global Judge). Твоя задача — оценить УСПЕХ всей сессии Process Mining.\n"
            "ПРАВИЛА:\n"
            "1. Изучи 'Long-Term Memory' (историю шагов) и 'Final Report'.\n"
            "2. КРИТЕРИИ УСПЕХА:\n"
            "   - Пройдены ли шаги: Profiling, Cleaning, Discovery, Visualization, Analysis?\n"
            "   - Есть ли финальный отчет?\n"
            "   - Есть ли ссылки на графики (визуализацию)?\n"
            "3. Если все хорошо — верни passed=True.\n"
            "4. Если есть КРИТИЧЕСКИЕ пробелы (например, не построены графики, или отчет пустой) — верни passed=False и КРИТИКУ.\n"
            "5. Верни JSON: {'passed': bool, 'critique': str, 'suggested_start_point': str (с чего начать исправление?)}\n"
        )
        prompt = f"Long-Term Memory:\n{memory}\n\nFinal Report:\n{final_report}"
        
        response = self.generate_response(prompt, system_prompt)
        
        try:
            import json
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
            return {"passed": True, "critique": "Parsing error", "suggested_start_point": ""}
        except:
             return {"passed": True, "critique": "Parsing error", "suggested_start_point": ""}


    def reflect_on_result(self, context: str, result: str) -> dict:
        # Legacy reflection method, kept for compatibility but Judge is preferred now
        return self.judge_step("Reflection", context, result)

    def update_memory(self, current_memory: str, step_name: str, step_result: str) -> str:
        """
        Updates the long-term memory with the result of a step.
        """
        system_prompt = (
            "Ты — Менеджер Памяти агента Process Mining. Твоя задача — поддерживать актуальное и сжатое состояние процесса (Memory).\n"
            "ПРАВИЛА:\n"
            "1. ЧИТАЙ 'Current Memory' и 'Latest Tool Output'.\n"
            "2. ДОБАВЬ новую информацию из Output в Memory. Сохраняй ТОЛЬКО факты: статус шага (Success/Fail), ключевые цифры (кол-во строк, статистику), имена созданных файлов.\n"
            "3. ВАЖНО: Если агент выявил ИНСАЙТЫ, АНОМАЛИИ, УЗКИЕ МЕСТА или предупреждения — ОБЯЗАТЕЛЬНО сохрани их! Это нужно для консультации пользователя в конце.\n"
            "4. УДАЛЯЙ устаревшие детали. Если была ошибка, но потом агент исправился — ошибку можно сократить до 'были проблемы, исправлено'.\n"
            "5. НЕ копируй полные логи. Будь краток. Используй Markdown списки.\n"
            "6. ОБЯЗАТЕЛЬНО сохраняй полный путь к каждому созданному файлу (report, image).\n"
            "Верни ОБНОВЛЕННЫЙ текст Memory."
        )
        prompt = f"Current Memory:\n{current_memory}\n\nLatest Tool ({step_name}) Output:\n{step_result}"
        return self.generate_response(prompt, system_prompt)

    def answer_user_question(self, memory: str, final_report: str, chat_history: str, question: str, knowledge_base: str = "", tools_desc: str = "") -> dict:
        """
        Answers user questions based on memory, report, chat history, KNOWLEDGE BASE, and optionally using TOOLS.
        Returns a JSON with 'answer', optional 'knowledge_update', and optional 'tool_call'.
        """
        import json
        
        tools_section = ""
        tools_list_for_capabilities = ""
        if tools_desc:
            tools_section = (
                f"\n\n=== ДОСТУПНЫЕ ИНСТРУМЕНТЫ АНАЛИЗА ===\n{tools_desc}\n"
                "**run_complex_analysis** (description='Use ONLY for complex queries that standard tools cannot handle. E.g. complex filtering, combining multiple metrics, advanced grouping.')\n"
                "=== КОНЕЦ СПИСКА ИНСТРУМЕНТОВ ===\n\n"
                "ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ:\n"
                "- Если вопрос ТРИВИАЛЬНЫЙ (частота активностей, длительность кейсов) — используй стандартные инструменты.\n"
                "- Если вопрос СЛОЖНЫЙ (фильтр 'начинается с X и заканчивается Y', 'средняя длительность по группе Z', 'медиана', 'перцентиль') — СРАЗУ используй `run_complex_analysis`!\n"
                "  (Аргументы для `run_complex_analysis`: можно передать пустой JSON {}).\n"
                "- ВНИМАТЕЛЬНО извлекай параметры из вопроса! Если спрашивают 'топ 10', 'последние 3' — передай это в аргументы (top_n=10).\n"
                "- Когда используешь tool_call, поле answer оставь пустым (null).\n"
            )
            tools_list_for_capabilities = tools_desc
        
        # Human-friendly capabilities description
        capabilities_text = (
            "ЕСЛИ СПРАШИВАЮТ 'ЧТО ТЫ УМЕЕШЬ' — ответь ПОНЯТНЫМ языком:\n"
            "Вот что я могу:\n"
            "📊 **Отвечать на вопросы по твоим данным** — расскажу про найденные аномалии, узкие места, инсайты из анализа.\n"
            "🔢 **Считать статистику** — средние, медианы, проценты, корреляции между любыми колонками.\n"
            "🔍 **Фильтровать данные** — покажу только нужные записи по условию.\n"
            "📈 **Анализировать процессы** — частота путей, длительность кейсов, самые частые активности.\n"
            "🚧 **Искать проблемы** — узкие места, выбросы, аномальные кейсы.\n"
            "💾 **Запоминать важное** — если скажешь 'запомни это', сохраню в базу знаний.\n"
            "Просто спрашивай на обычном языке!\n\n"
        )
        
        system_prompt = (
            "Ты — Эксперт-консультант по Process Mining с доступом к инструментам анализа данных.\n"
            "\n"
            "ИСТОЧНИКИ ДАННЫХ:\n"
            "1. MEMORY — хронология действий агента.\n"
            "2. FINAL REPORT — итоговый отчет с метриками.\n"
            "3. KNOWLEDGE BASE — глоссарий и факты от пользователя.\n"
            "4. CHAT HISTORY — история диалога.\n"
            + tools_section +
            "\n"
            + capabilities_text +
            "ПРАВИЛА:\n"
            "1. Если CHAT HISTORY пуст — пользователь ещё ничего не спрашивал. НЕ выдумывай.\n"
            "2. Если пользователь просит запомнить важное — сохрани в knowledge_update.\n"
            "3. Если нужны РАСЧЕТЫ — используй tool_call, НЕ выдумывай цифры.\n"
            "4. Отвечай на языке вопроса (русский), простым понятным языком.\n"
            "\n"
            "ФОРМАТ ОТВЕТА — СТРОГО JSON:\n"
            "```json\n"
            "{\n"
            '  "answer": "Текст ответа (или null если вызываешь инструмент)",\n'
            '  "knowledge_update": "Важный факт для сохранения (или null)",\n'
            '  "tool_call": {"name": "имя_инструмента", "args": {"arg1": "val1"}} или null\n'
            "}\n"
            "```\n"
            "ВАЖНО: Возвращай ТОЛЬКО JSON, без лишнего текста. Если вызываешь tool_call, answer ДОЛЖЕН быть null."
        )
        
        user_prompt = (
            f"KNOWLEDGE BASE:\n{knowledge_base}\n\n"
            f"MEMORY:\n{memory}\n\n"
            f"FINAL REPORT:\n{final_report}\n\n"
            f"CHAT HISTORY:\n{chat_history}\n\n"
            f"USER QUESTION:\n{question}\n"
        )
        
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        # Clean markdown if present
        cleaned_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Fallback if model fails JSON
            return {"answer": response_str, "knowledge_update": None, "tool_call": None}

    def interpret_tool_result(self, question: str, tool_result: dict) -> dict:
        """
        Interprets tool result into a human-friendly answer.
        """
        import json
        system_prompt = (
            "Ты — Эксперт-консультант по Process Mining.\n"
            "Пользователь задал вопрос, и был вызван инструмент анализа.\n"
            "Твоя задача — превратить результат инструмента в понятный ответ.\n"
            "\n"
            "ВАЖНО: НИКОГДА НЕ ВЫДУМЫВАЙ ЦИФРЫ.\n"
            "- Если в результате НЕТ нужных данных (например, спросили про 10-й элемент, а их всего 5) -> "
            "честно скажи: 'Инструмент вернул только 5 записей, я не вижу 10-ю'. НЕ пытайся угадать.\n"
            "- Используй ТОЛЬКО факты из JSON.\n"
            "\n"
            "ФОРМАТ ВЫХОДА (JSON):\n"
            '{"answer": "Человекочитаемый ответ..."}'
        )
        
        user_prompt = (
            f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}\n\n"
            f"РЕЗУЛЬТАТ ИНСТРУМЕНТА:\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n"
        )
        
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        cleaned_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            return {"answer": str(tool_result)}

    def generate_pandas_code(self, question: str, df_info: str, previous_error: str = "", context: dict = None) -> dict:
        """
        Generates pandas code to answer the user's question, using full context.
        context keys: 'knowledge_base', 'memory', 'final_report'
        """
        import json
        
        context = context or {}
        knowledge_base = context.get("knowledge_base", "")
        memory = context.get("memory", "")
        final_report = context.get("final_report", "")
        
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
            "\n"
            "ТЕРМИНОЛОГИЯ PROCESS MINING:\n"
            "- АКТИВНОСТЬ (Activity, Operation) = Событие, строка в логе. Частота активностей = df['activity_col'].value_counts().\n"
            "- ПУТЬ/ТРЕЙС (Path, Trace, Variant) = ПОСЛЕДОВАТЕЛЬНОСТЬ активностей для одного case_id.\n"
            "  ВАЖНО: Для анализа путей, собери их в СТРОКУ через разделитель: .apply(lambda x: ' -> '.join(x)).\n"
            "  НЕ работай со списками или кортежами в индексах (value_counts на списках вызовет ошибку!).\n"
            "\n"
            "PANDAS BEST PRACTICES (ЧТОБЫ ИЗБЕЖАТЬ ОШИБОК):\n"
            "1. `value_counts()` возвращает Series. У неё НЕТ `.to_dict()` для строки. Чтобы получить словарь {{index: ..., count: ...}}, используй `.reset_index().iloc[i].to_dict()`.\n"
            "   - ПЛОХО: `vc.iloc[0].to_dict()` (AttributeError)\n"
            "   - ХОРОШО: `vc.reset_index().iloc[0].to_dict()`\n"
            "2. ПРОВЕРЯЙ ГРАНИЦЫ ИНДЕКСА! Если просят 10000-й элемент, проверь `len(df) > 9999`.\n"
            "   - `idx = 9999; result = vc.index[idx] if len(vc) > idx else 'Элемент не найден'`\n"
            "3. `.iloc[i]` возвращает скаляр (numpy type). Используй `.item()` чтобы сделать его Python-типом.\n"
            "4. ВРЕМЯ (Duration): Перед расчетом времени ВСЕГДА делай `.sort_values('timestamp')`. Иначе получишь отрицательное время!\n"
            "5. ФИЛЬТР ПО ПУТИ (Starts/Ends with): НЕ используй `df[df.col.isin(...)]` — это ломает порядок. Правильно: `df.groupby(case).filter(lambda x: x.iloc[0]==Start and x.iloc[-1]==End)`.\n"
            "\n"
            "ПРАВИЛА:\n"
            "1. ОБЯЗАТЕЛЬНО сохрани результат в переменную 'result'.\n"
            "2. Код должен быть простым и читаемым.\n"
            "3. Используй только pandas/numpy операции, никаких import.\n"
            "4. ВАЖНО: 'result' должен быть стандартным Python типом (int, float, dict, list), а НЕ numpy.int64. Используй .item() для скаляров.\n"
            "4. Если нужна группировка по кейсам, используй колонку с ID кейса.\n"
            "5. Результат должен быть JSON-сериализуемым (числа, строки, списки, dict).\n"
            + error_context +
            "\n"
            "ФОРМАТ ОТВЕТА (JSON):\n"
            "{\n"
            '  "thought": "Рассуждение: что нужно сделать и как",\n'
            '  "code": "result = df..."\n'
            "}"
        )
        
        user_prompt = (
            f"KNOWLEDGE BASE:\n{knowledge_base}\n\n"
            f"MEMORY (ПРЕДЫДУЩИЙ КОНТЕКСТ):\n{memory}\n(Используй Memory, если вопрос ссылается на 'предыдущий' или 'такой же' фильтр!)\n\n"
            f"FINAL REPORT:\n{final_report}\n\n"
            f"ИНФОРМАЦИЯ О ДАННЫХ:\n{df_info}\n\n"
            f"ВОПРОС:\n{question}"
            f"{error_context}"
        )
        
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        # Fix markdown stripping
        cleaned_str = response_str.strip()
        if "```json" in cleaned_str:
             cleaned_str = cleaned_str.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned_str:
             cleaned_str = cleaned_str.split("```")[0].strip()
        
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Fallback: try to find ANYTHING that looks like JSON
            import re
            json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            # Last resort
            return {"thought": "Не удалось распарсить JSON", "code": response_str}

    def verify_result(self, question: str, result_str: str) -> dict:
        """
        Verifies if the result adequately answers the question.
        Returns: {"is_valid": bool, "critique": str, "suggestion": str}
        """
        import json
        system_prompt = (
            "Ты — строгий критик. Проверь, содержит ли результат ВСЕ необходимые данные для ответа.\n"
            "НЕ пересчитывай цифры, проверяй ПОЛНОТУ данных.\n"
            "\n"
            "КРИТИЧЕСКИЕ ОШИБКИ (is_valid: false):\n"
            "- Пользователь спросил 'Топ 10', а в списке меньше 10 элементов. (НО: Если спросили '10-й элемент' и вернули ТОЛЬКО ЕГО — это ВЕРНО! Не требуй всех 10, если нужен только один).\n"
            "- Пользователь спросил конкретное число, а его нет в ответе инструмента.\n"
            "- Результат пустой или содержит ошибку (например, 'Column does not exist').\n"
            "- Ответ 'Я не знаю' или 'Данных нет' — это допустимо, если данных действительно нет, но если их можно получить — дай совет.\n"
            "- Использован НЕВЕРНЫЙ инструмент (например, сортировка строк вместо частоты путей).\n"
            "- ВЫЧИСЛЕНА НЕ ТА МЕТРИКА (например, посчитали частоту АКТИВНОСТЕЙ, а просили частоту ПУТЕЙ).\n"
            "  * ПУТЬ (Trace) = последовательность активностей для одного кейса. Обычно выглядит как 'A -> B -> C'.\n"
            "  * АКТИВНОСТЬ (Activity) = одно событие.\n"
            "\n"
            "ЧАСТИЧНЫЙ УСПЕХ (is_valid: 'partial'):\n"
            "- Ответ В ЦЕЛОМ ВЕРНЫЙ, но есть мелкие недочеты формата (например, просили проценты, а дали только число).\n"
            "- Ответ СОДЕРЖИТ нужную цифру, но с лишним 'мусором'.\n"
            "- Инструмент вернул ТОЛЬКО 5 записей вместо 10 (как просили), но это лучше чем ничего.\n"
            "В таких случаях возвращай 'partial', чтобы мы показали пользователю то, что есть.\n"
            "\n"
            "ФОРМАТ ОТВЕТА (JSON):\n"
            "{\n"
            '  "thought": "Ответ верный, но формат не совсем тот...",\n'
            '  "is_valid": true | false | "partial",\n'
            '  "critique": "...",\n'
            '  "suggestion": "..."\n'
            "}"
            "Если данных достаточно и логика верна, верни is_valid: true."
        )
        
        user_prompt = f"ВОПРОС: {question}\n\nРЕЗУЛЬТАТ: {result_str}"
        
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        # Handle markdown if present
        cleaned_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(cleaned_str)
        except:
            return {
                "is_valid": False, 
                "thought": f"Ошибка парсинга ответа проверки (JSON error). Raw: {response_str}",
                "critique": "Не удалось проверить результат (сбой JSON).",
                "suggestion": "Попробуй выполнить код еще раз."
            }

    def interpret_code_result(self, question: str, result: str, result_type: str) -> dict:
        """
        Interprets code execution result into a human-friendly answer.
        """
        import json
        system_prompt = (
            "Ты — Эксперт-консультант по Process Mining.\n"
            "Пользователь задал вопрос, и был выполнен pandas-код.\n"
            "Твоя задача — превратить результат в понятный ответ.\n"
            "ПРАВИЛА:\n"
            "- Отвечай кратко, на русском языке.\n"
            "- Используй цифры/факты из результата.\n"
            "- Если результат — это ДЛИННЫЙ ПУТЬ (строка A -> B -> ...), покажи его ПЕЛИКОМ (или начло...конец), но НЕ сокращай до одного слова!\n"
            "- Не выдумывай названия процессов, цитируй прямо из данных.\n"
            "\n"
            "ФОРМАТ (JSON):\n"
            '{"answer": "Ответ..."}'
        )
        
        user_prompt = f"ВОПРОС:\n{question}\n\nРЕЗУЛЬТАТ ({result_type}):\n{result}"
        
        response_str = self.generate_response(user_prompt, system_prompt, json_mode=True)
        cleaned_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            return {"answer": result}

    def decide_next_step(self, memory: str, tools_description: str) -> dict:
        """
        Analyzes the current memory and decides the next step (tool to use).
        Returns a JSON with 'thought' and 'tool_name'.
        """
        system_prompt = (
            "Ты — умный оркестратор агента Process Mining (AutoPM). Твоя цель — провести полный анализ процесса от загрузки до финального отчета.\n"
            "У тебя есть набор инструментов (агентов). Твоя задача — рассуждать и выбирать следующий шаг, основываясь на ДОЛГОСРОЧНОЙ ПАМЯТИ.\n\n"
            "ПРАВИЛА:\n"
            "1. Анализируй 'Memory' (текущее состояние). Если шаг отмечен как DONE/Success, НЕ ПОВТОРЯЙ его, переходи к следующему логическому шагу.\n"
            "2. ЛОГИЧЕСКАЯ ЦЕПОЧКА ПО УМОЛЧАНИЮ: Data Profiling -> Data Cleaning -> Process Discovery -> Visualization -> Process Analysis -> Reporting -> Finish.\n"
            "   - Visualization и Process Discovery независимы, их порядок можно менять, но обычно Visualization идет раньше.\n"
            "   - Reporting ВСЕГДА последний перед Finish.\n"
            "3. Если в Памяти есть активная проблема или ОШИБКА последнего шага, выбери инструмент для её исправления (или повтори шаг).\n"
            "4. ОТВЕТ ДОЛЖЕН БЫТЬ В ФОРМАТЕ JSON:\n"
            "   {\n"
            "     \"thought\": \"Твое рассуждение. Что мы уже сделали (согласно Memory)? Что нужно сделать дальше?\",\n"
            "     \"tool_name\": \"Название инструмента из списка Available Tools (или 'Finish', если все готово)\"\n"
            "   }\n"
        )
        
        prompt = f"Long-Term Memory:\n{memory}\n\nAvailable Tools:\n{tools_description}\n\nКакой следующий шаг?"
        
        response = self.generate_response(prompt, system_prompt)
        
        try:
            import json
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(response[start:end])
            # Fallback for bad LLM output
            return {"thought": f"Failed to parse JSON. Raw response: {response}", "tool_name": "Reporting"} 
        except Exception as e:
            return {"thought": f"Error parsing logic: {e}", "tool_name": "Final Report"}
