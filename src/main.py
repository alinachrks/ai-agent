import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from data_analysis import load_data, exploratory_data_analysis
from model_training import train_model
from code_updater import update_code
from commit_message import generate_commit_message

# Загрузка и предобработка данных с учетом гибкости
def load_and_preprocess_data(dataset_id):
    # Загрузка набора данных из OpenML (по ID)
    dataset = openml.datasets.get_dataset(dataset_id)
    df, *_ = dataset.get_data()

    # Просмотр первых нескольких строк для понимания структуры
    print(df.head())

    # Автоматическая обработка категориальных данных
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Разделяем данные на признаки и целевую переменную
    target_column = 'Class' if 'Class' in df.columns else df.columns[-1]  # Если 'Class' нет, берем последнюю колонку как целевую
    features = df.drop(target_column, axis=1)  
    target = df[target_column]

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dataset_id = 1464
    
    # Загружаем и предобрабатываем данные
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_id)

    # Обучаем модель
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Оценка модели
    y_pred = model.predict(X_test)

    # Генерация отчета о классификации
    report = classification_report(y_test, y_pred)
    with open("report.md", "w") as f:
        f.write("## Отчет по классификации\n")
        f.write(f"**Модель:** RandomForestClassifier\n")
        f.write(f"**Оценка модели:**\n{report}\n")

    # Запуск функции обновления кода
    update_code()

    # Генерация сообщения для коммита
    commit_msg = generate_commit_message(changes_summary="Обновление модели для анализа данных OpenML")

    print(f"✅ Обновление завершено. Название коммита: {commit_msg}")
