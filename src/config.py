import os

# API-ключ для Google Gemini (загружается из переменных окружения)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Проверяем наличие API-ключа
if not GEMINI_API_KEY:
    raise ValueError("API-ключ для Gemini не найден! Добавьте его в GitHub Secrets.")
