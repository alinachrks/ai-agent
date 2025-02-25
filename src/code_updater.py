import requests
import os
from config import GEMINI_API_KEY

# Новый формат запроса к Gemini API
def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Проверка на успешный ответ

        # Обновленный формат ответа
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            output = data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return output if output else "Ошибка: пустой ответ от API."
        return "Ошибка: API не вернул кандидатов."
    
    except requests.exceptions.RequestException as e:
        return f"Ошибка при запросе к API: {e}"

# Функция для обновления кода модели
def update_code():
    file_path = "src/model_training.py"

    if not os.path.exists(file_path):
        print(f"Ошибка: файл {file_path} не найден.")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            current_code = f.read()

        prompt = f"""
        Вот код Python для обучения модели машинного обучения:
        {current_code}

        Улучшите этот код, добавьте новые функции и оптимизируйте его.
        """

        new_code = query_gemini(prompt)

        if "Ошибка" in new_code:
            print(new_code)
            return False

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_code)

        print("✅ Код успешно обновлён через Gemini API!")
        return True

    except Exception as e:
        print(f"❌ Ошибка обновления кода: {e}")
        return False

if __name__ == "__main__":
    update_code()

