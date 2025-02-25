import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Получение API-ключа Gemini из переменных среды
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Проверка наличия ключа
if not GEMINI_API_KEY:
    raise ValueError("API-ключ для Gemini не найден! Добавьте его в GitHub Secrets.")

# Функция для запроса к Gemini API
def query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.7,
        "max_tokens": 300
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()["candidates"][0]["output"]
    else:
        return f"Ошибка API: {response.text}"

# Функция загрузки данных (пример с датасетом Iris)
def load_data():
    df = sns.load_dataset("iris")
    return df

# Функция анализа данных (EDA)
def exploratory_data_analysis(df):
    report = f"""## Автоматический анализ данных
    
    **Размерность данных:** {df.shape}
    
    **Пропущенные значения:**\n{df.isnull().sum()}
    
    **Основные статистики:**\n{df.describe()}
    """

    # Генерация визуализации данных
    sns.pairplot(df, hue=df.columns[-1])
    plt.savefig("eda_plot.png")

    return report

# Функция обучения модели
def train_model(df):
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y.dtype == 'O':  # Классификация
        model = RandomForestClassifier()
        metric_func = accuracy_score
        metric_name = "Accuracy"
    else:  # Регрессия
        model = RandomForestRegressor()
        metric_func = mean_squared_error
        metric_name = "RMSE"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metric = metric_func(y_test, y_pred)

    return model, metric_name, metric

if __name__ == "__main__":
    df = load_data()
    eda_report = exploratory_data_analysis(df)
    model, metric_name, metric = train_model(df)

    # Формируем отчёт
    prompt_text = f"""
    Данные о наборе данных:
    {eda_report}

    Метрика модели: {metric_name} = {metric:.4f}

    Опиши выводы по данным и предложи, какие улучшения можно сделать в ML-модели.
    """
    gemini_response = query_gemini(prompt_text)

    # Сохранение отчёта
    with open("report.md", "w") as f:
        f.write(eda_report)
        f.write(f"\n\n**Метрика модели ({metric_name}): {metric:.4f}**\n\n")
        f.write(f"### 🤖 Вывод Gemini AI:\n{gemini_response}")

    print("Готово! Анализ данных и обучение модели завершены.")
