import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(dataset_id=1464):
    """
    Загружает и предобрабатывает данные из OpenML, автоматически кодируя категориальные переменные.
    """
    dataset = openml.datasets.get_dataset(dataset_id)
    df, *_ = dataset.get_data()

    print("🔹 Первые строки датасета:")
    print(df.head())

    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = label_encoder.fit_transform(df[column])

    target_column = "Class" if "Class" in df.columns else df.columns[-1]
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """
    Обучает модель RandomForestClassifier и возвращает отчет о её качестве.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    
    report_path = "report.md"
    with open(report_path, "w") as f:
        f.write("## Отчет по классификации\n")
        f.write(f"**Модель:** RandomForestClassifier\n")
        f.write(f"**Оценка модели:**\n{report}\n")

    print(f"✅ Отчет сохранен в {report_path}")
    return model, report

if __name__ == "__main__":
    dataset_id = 1464
    print("📥 Загрузка и предобработка данных...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_id)

    print("🎯 Обучение модели...")
    model, report = train_model(X_train, X_test, y_train, y_test)

    print("📊 Оценка модели завершена.")

