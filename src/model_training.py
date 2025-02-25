import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(dataset_id=1464):
    """
    Загружает и предобрабатывает данные из OpenML, автоматически кодируя категориальные переменные.
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        df, *_ = dataset.get_data()

        print("🔹 Первые строки датасета:")
        print(df.head())

        # Обрабатываем категориальные данные с LabelEncoder
        label_encoder = LabelEncoder()
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = label_encoder.fit_transform(df[column])

        # Определяем целевую переменную ('Class' или последняя колонка)
        target_column = "Class" if "Class" in df.columns else df.columns[-1]
        features = df.drop(target_column, axis=1)
        target = df[target_column]

        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    """
    Обучает модель RandomForestClassifier и возвращает обученную модель.
    """
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("✅ Модель успешно обучена.")
        return model
    except Exception as e:
        print(f"❌ Ошибка при обучении модели: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    """
    Оценивает модель и сохраняет отчёт.
    """
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)

        # Сохранение отчёта
        report_path = "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("## Отчет по классификации\n")
            f.write(f"**Модель:** RandomForestClassifier\n")
            f.write(f"**Оценка модели:**\n{report}\n")

        print(f"✅ Отчёт сохранён в {report_path}")
    except Exception as e:
        print(f"❌ Ошибка при оценке модели: {e}")

if __name__ == "__main__":
    # Загружаем и предобрабатываем данные
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    if X_train is not None and y_train is not None:
        # Обучаем модель
        model = train_model(X_train, y_train)

        if model is not None:
            # Оцениваем модель
            evaluate_model(model, X_test, y_test)
