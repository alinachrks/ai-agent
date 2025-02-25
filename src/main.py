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

# Загрузка и предобработка данных
def load_and_preprocess_data():
    # Загрузка набора данных из OpenML (ID 1464)
    dataset = openml.datasets.get_dataset(1464)
    df, *_ = dataset.get_data()

    # Просмотр первых нескольких строк для понимания структуры
    print(df.head())

    # Преобразуем категориальные данные в числовые с помощью LabelEncoder
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Разделяем данные на признаки и целевую переменную (например, прогнозирование класса)
    features = df.drop('Class', axis=1)  # Предполагаем, что 'Class' - целевая переменная
    target = df['Class']

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Загружаем и предобрабатываем данные
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Обучаем модель RandomForest
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
