import requests
from config import GEMINI_API_KEY

# Функция для запроса к Gemini API
def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": {"text": prompt}, "temperature": 0.7, "max_tokens": 500}

    try:
        # Отправка запроса к API
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Проверка на успешный ответ

        # Извлечение текста ответа
        output = response.json().get("candidates", [{}])[0].get("output", "")
        
        if output:
            return output
        else:
            return "Ответ не содержит текста."
    except requests.exceptions.RequestException as e:
        # Обработка ошибок запроса
        return f"Ошибка при запросе к API: {e}"

# Функция для обновления кода модели
def update_code():
    try:
        # Читаем текущий код из файла
        with open("src/model_training.py", "r") as f:
            current_code = f.read()

        # Формируем запрос для улучшения кода
        prompt = f"""
        Вот код Python, отвечающий за обучение ML-модели:
        {current_code}

        Улучшите этот код, добавьте новые функции или оптимизации.
        """

        # Получаем новый код от API
        new_code = query_gemini(prompt)

        # Проверка на ошибку API
        if "Ошибка при запросе к API" in new_code:
            print(new_code)
            return False

        # Записываем улучшенный код обратно в файл
        with open("src/model_training.py", "w") as f:
            f.write(new_code)

        print("Код был успешно обновлен.")
        return True

    except FileNotFoundError:
        # Ошибка, если файл не найден
        print("Ошибка: файл src/model_training.py не найден.")
        return False
    except Exception as e:
        # Обработка других исключений
        print(f"Неизвестная ошибка: {e}")
        return False

if __name__ == "__main__":
    # Вызываем обновление кода
    if update_code():
        print("Процесс завершен успешно.")
    else:
        print("Процесс завершен с ошибкой.")
