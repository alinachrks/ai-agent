import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Импортируем локальные модули
from data_analysis import load_data, exploratory_data_analysis
from model_training import train_model
from code_updater import update_code
from commit_message import generate_commit_message


def load_and_preprocess_data(dataset_id):
    """
    Загружает и предобрабатывает данные из OpenML, автоматически кодируя категориальные переменные.
    """
    # Загружаем набор данных
    dataset = openml.datasets.get_dataset(dataset_id)
    df, *_ = dataset.get_data()

    # Просмотр первых строк для понимания структуры
    print("🔹 Первые строки датасета:")
    print(df.head())

    # Обрабатываем категориальные данные с LabelEncoder
    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Определяем целевую переменную (если 'Class' нет, берём последнюю колонку)
    target_column = "Class" if "Class" in df.columns else df.columns[-1]
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    # Разделяем на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    dataset_id = 1464  # ID набора данных OpenML

    print("📥 Загрузка и предобработка данных...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_id)

    print("🎯 Обучение модели...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("📊 Оценка модели...")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    # Сохранение отчёта
    report_path = "report.md"
    with open(report_path, "w") as f:
        f.write("## Отчет по классификации\n")
        f.write(f"**Модель:** RandomForestClassifier\n")
        f.write(f"**Оценка модели:**\n{report}\n")

    print(f"✅ Отчет сохранен в {report_path}")

    # Запускаем обновление кода через Gemini API
    print("🔄 Обновление кода через Gemini...")
    update_code()

    # Генерируем сообщение для коммита
    commit_msg = generate_commit_message(changes_summary="Обновление модели для анализа данных OpenML")

    print(f"✅ Обновление завершено. Название коммита: {commit_msg}")

