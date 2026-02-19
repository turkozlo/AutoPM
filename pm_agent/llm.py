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
        else:
            self.client = ChatMistralAI(
                model_name=MISTRAL_MODEL, api_key=MISTRAL_API_KEY, temperature=0.2
            )

    def _parse_json(self, response_str: str) -> dict:
        """Helper for internal needs (formatter.py still uses JSON)."""
        import json
        import re
        if not response_str: return None
        try: return json.loads(response_str)
        except: pass
        match = re.search(r"(\{.*\})", response_str, re.DOTALL)
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

    def simple_chat(self, user_query: str, context: str) -> str:
        """
        Simple chat that only knows basic dataset stats.
        """
        system_prompt = (
            "Ты — строгий ассистент, который знает ТОЛЬКО базовую информацию о датасете (названия колонок, количество строк и столбцов).\n"
            "ПРАВИЛА:\n"
            "1. Отвечай на вопросы, касающиеся структуры данных (сколько строк, какие колонки).\n"
            "2. Если пользователь просит посчитать среднее, медиану, построить график или выполнить код — ОТКАЗЫВАЙСЯ.\n"
            "3. Скажи вежливо: 'Я пока умею только читать структуру файла (строки, колонки), но не умею считать сложную статистику'.\n"
            "4. Используй предоставленный КОНТЕКСТ для ответов.\n"
            "5. Отвечай на русском языке."
        )
        prompt = f"КОНТЕКСТ ДАННЫХ:\n{context}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{user_query}"
        
        return self.generate_response(prompt, system_prompt)
