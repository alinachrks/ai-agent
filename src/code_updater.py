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
        response_json = response.json()
        candidates = response_json.get("candidates", [])
        if candidates:
            return candidates[0].get("output", "")
        else:
            return "Ошибка API: Gemini не вернул кандидатов."
    except requests.exceptions.RequestException as e:
        return f"Ошибка API: {e}"

# Функция для улучшения кода обучения модели
def update_code():
    file_path = "src/model_training.py"
    
    if not os.path.exists(file_path):
        print(f"Ошибка: файл {file_path} не найден.")
        return False
    
    try:
        # Читаем текущий код из файла
        with open(file_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        # Промт для запроса к Gemini API с улучшенной формулировкой
        prompt = f"""
        Вы являетесь экспертом в области машинного обучения и разработки программного обеспечения.
        Ваша задача - улучшить код Python, отвечающий за обучение модели машинного обучения.
        Код может включать в себя:
        - Оптимизацию производительности
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

        if "Ошибка API" in new_code or not new_code.strip().startswith("import"):
            print(f"Ошибка при получении ответа от Gemini: {new_code}")
            return False

        # Записываем улучшенный код в файл
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        print("✅ Код был успешно обновлен с улучшениями и новыми визуализациями.")
        return True

    except Exception as e:
        print(f"❌ Неизвестная ошибка: {e}")
        return False

if __name__ == "__main__":
    update_code()
