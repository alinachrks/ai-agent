import requests
from config import GEMINI_API_KEY

def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": {"text": prompt}, "temperature": 0.7, "max_tokens": 500}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Проверка на успешный ответ
        return response.json()["candidates"][0]["output"]
    except requests.exceptions.RequestException as e:
        return f"Ошибка при запросе к API: {e}"

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
