import requests
import os
from config import GEMINI_API_KEY

# Функция для запроса к Gemini API
def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "max_tokens": 1000  # Увеличение количества токенов для более детализированных ответов
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Проверка на успешный ответ
        return response.json()["candidates"][0]["output"]
    except requests.exceptions.RequestException as e:
        return f"Ошибка при запросе к API: {e}"

# Функция для улучшения кода обучения модели
def update_code():
    try:
        # Читаем текущий код из файла
        with open("src/model_training.py", "r") as f:
            current_code = f.read()

        # Промт для запроса к Gemini API с улучшенной формулировкой
        prompt = f"""
        Вы являетесь экспертом в области машинного обучения и разработки программного обеспечения.
        Вашей задачей является улучшить код Python, который отвечает за обучение модели машинного обучения.
        Код может включать в себя любые аспекты:
        - Оптимизация производительности
        - Улучшение читаемости и поддерживаемости
        - Добавление новых полезных функций или методов
        - Применение современных методов и библиотек

        Также, пожалуйста:
        1. Добавьте новые визуализации данных, чтобы лучше понять тренды и поведение модели.
        2. Напишите подробное описание для каждой визуализации, объяснив, что она показывает.
        3. Увеличьте объём данных, используя методы, такие как расширение выборки или добавление синтетических данных.
        4. Укажите слабые стороны текущего анализа и проблематику, которую стоит учесть при дальнейшем улучшении модели.

        Вот текущий код:
        {current_code}

        Пожалуйста, улучшите этот код, добавьте новые функции, визуализации, расширьте данные и опишите слабые места анализа.
        """

        new_code = query_gemini(prompt)

        if "Ошибка API" in new_code:
            print(f"Ошибка при получении ответа от Gemini: {new_code}")
            return False

        # Записываем улучшенный код в файл
        with open("src/model_training.py", "w") as f:
            f.write(new_code)

        print("Код был успешно обновлен с улучшениями и новыми визуализациями.")
        return True

    except FileNotFoundError:
        print("Ошибка: файл src/model_training.py не найден.")
        return False
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return False

if __name__ == "__main__":
    update_code()
