import requests
import os
from config import GEMINI_API_KEY

# Функция для запроса к Gemini API
def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": {"text": prompt}, "temperature": 0.7, "max_tokens": 500}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["candidates"][0]["output"]
    else:
        return f"Ошибка API: {response.text}"

# Обновление `model_training.py`
def update_code():
    with open("src/model_training.py", "r") as f:
        current_code = f.read()

    prompt = f"""
    Вот код Python, отвечающий за обучение ML-модели:
    {current_code}

    Улучшите этот код, добавьте новые функции или оптимизации.
    """
    
    new_code = query_gemini(prompt)

    if "Ошибка API" in new_code:
        print(new_code)
        return False

    with open("src/model_training.py", "w") as f:
        f.write(new_code)

    return True

if __name__ == "__main__":
    update_code()
