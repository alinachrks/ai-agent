import requests
import os
from config import GEMINI_API_KEY

def generate_commit_message():
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": {"text": "Сформулируй название коммита для обновления Python-кода. Будь краток."}, "temperature": 0.7, "max_tokens": 20}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["candidates"][0]["output"]
    else:
        return "Автоматическое обновление кода"

if __name__ == "__main__":
    print(generate_commit_message())
