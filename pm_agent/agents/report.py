import json
from typing import Dict, Any

class ReportAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, artifacts: Dict[str, Any]) -> str:
        """
        Generates a final report based on all artifacts.
        artifacts: dict containing 'profiling', 'cleaning', 'visualization', 'discovery', 'analysis'.
        """
        system_prompt = (
            "Ты — Report Agent. Твоя задача — составить финальный аналитический отчет на основе предоставленных JSON-данных. "
            "Отчет должен быть структурированным, фактологическим и написан на русском языке. "
            "Используй Markdown. "
            "ПРАВИЛА: "
            "1. ОБЯЗАТЕЛЬНО используй конкретные цифры и ЕДИНИЦЫ ИЗМЕРЕНИЯ (сек., мин., час., дн.) из JSON. "
            "2. ОБЯЗАТЕЛЬНО вставляй ссылки на изображения в формате ![Описание](путь_к_файлу). "
            "3. ОБЯЗАТЕЛЬНО вставь Mermaid-код из раздела DISCOVERY в блок ```mermaid. "
            "   ВАЖНО: Копируй ТЕКСТ из поля 'mermaid'. Если в JSON ты видишь обратные слэши перед кавычками (например, \\\"), НЕ КОПИРУЙ ИХ. "
            "   В отчете должен быть чистый Mermaid-код. "
            "   ЗАПРЕЩЕНО придумывать свою схему. "
            "   Если поле 'mermaid' в JSON пустое, напиши 'Схема процесса не может быть построена'. "
            "   Пример: "
            "   ```mermaid"
            "   graph TD"
            "   node1[\"Название\"] --> node2[\"Другое\"]"
            "   ```"
            "4. ЗАПРЕЩЕНО делать выводы, не подкрепленные цифрами из JSON. "
            "5. ЗАПРЕЩЕНО менять единицы измерения, если в JSON указаны 'мин.', пиши 'мин.', не переводи в часы самостоятельно. "
            "6. Структура: "
            "   - # Аналитический отчет по процессу "
            "   - ## 1. Обзор данных (Profiling): количество строк, колонок, пропуски, дубликаты. "
            "   - ## 2. Очистка данных (Cleaning): сколько удалено, что заполнено. "
            "   - ## 3. Визуальный анализ (Visualization): интерпретация графиков со ссылками на файлы. "
            "   - ## 4. Модель процесса (Discovery): количество активностей, переходов, циклов и ИНТЕРАКТИВНАЯ СХЕМА (Mermaid). "
            "   - ## 5. Производительность (Analysis): среднее время, узкие места, аномалии. "
            "   - ## 6. Заключение и рекомендации. "
        )
        
        # Prepare context
        context = ""
        for section, data in artifacts.items():
            context += f"\n--- {section.upper()} ---\n"
            if isinstance(data, str):
                context += data
            else:
                context += json.dumps(data, indent=2, ensure_ascii=False)
        
        prompt = f"Сгенерируй отчет на основе этих данных:\n{context}"
        
        report = self.llm.generate_response(prompt, system_prompt)
        return report

