import requests
import os
from config import GEMINI_API_KEY

# Функция для генерации сообщения коммита с улучшенным запросом
def generate_commit_message(changes_summary=""):
    """
    Генерирует сообщение для коммита, основываясь на описании изменений в коде.
    Если описание изменений не предоставлено, используется стандартное сообщение.
    
    :param changes_summary: Строка с кратким описанием изменений, которые были внесены в код.
    :return: Сообщение для коммита
    """
    # Формулировка запроса для Gemini API с дополнительным контекстом
    prompt = f"""
    Создай название для коммита для обновления Python-кода, с учётом следующих изменений:
    {changes_summary if changes_summary else "Обновления кода"}

    Будь кратким, но точным, и постарайся отразить суть изменений.
    """

    # Запрос к API
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": {"text": prompt}, "temperature": 0.7, "max_tokens": 30}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Проверка на успешный ответ

        # Возвращаем сформированное сообщение для коммита
        return response.json()["candidates"][0]["output"]

    except requests.exceptions.RequestException as e:
        # Обработка ошибок запросов
        print(f"Ошибка при запросе к Gemini API: {e}")
        return "Автоматическое обновление кода"

if __name__ == "__main__":
    # Пример использования функции с описанием изменений
    changes = "Добавлены новые визуализации и улучшены алгоритмы анализа данных"
    print(generate_commit_message(changes))
