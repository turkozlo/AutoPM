# Local LLM Server

OpenAI-совместимый сервер для локального запуска LLM через HuggingFace Transformers.

## Установка

```bash
cd llm_server
pip install -r requirements.txt
```

## Запуск

```bash
python server.py --model ../models/Qwen_Qwen2.5.14B-Instruct --port 5000
```

Или с явным указанием хоста:

```bash
python server.py --model ../models/Qwen_Qwen2.5.14B-Instruct --port 5000 --host 0.0.0.0
```

## Проверка

```bash
# Health check
curl http://localhost:5000/health

# Список моделей
curl http://localhost:5000/v1/models

# Генерация ответа
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Привет!"}],
    "max_tokens": 256
  }'
```

## Подключение из AutoPM

В `config.yaml` укажите:

```yaml
provider: "local"
local:
  base_url: "http://localhost:5000/v1"
  model: "models/Qwen_Qwen2.5.14B-Instruct"
  api_key: "123"
```

## Важно

- Сервер работает **полностью оффлайн** (`HF_HUB_OFFLINE=1`).
- Модель должна быть **предварительно скачана** в локальную папку.
- При наличии GPU используется автоматически (float16 + `device_map=auto`).
