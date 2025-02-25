from model_training import load_and_preprocess_data, train_model, evaluate_model
from data_analysis import exploratory_data_analysis
from code_updater import update_code
from commit_message import generate_commit_message

def main():
    """
    Основной процесс:
    1. Загрузка и предобработка данных
    2. Анализ данных (EDA)
    3. Обучение модели
    4. Оценка модели
    5. Автоматическое обновление кода
    6. Генерация сообщения для коммита
    """
    dataset_id = 1464  # ID набора данных OpenML

    print("📥 Загружаем и предобрабатываем данные...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_id)

    if X_train is None or y_train is None:
        print("❌ Ошибка загрузки данных. Завершение работы.")
        return

    print("📊 Выполняем анализ данных (EDA)...")
    exploratory_data_analysis(X_train, y_train)  # Генерируем графики и статистику

    print("🎯 Обучаем модель...")
    model = train_model(X_train, y_train)

    if model is None:
        print("❌ Ошибка обучения модели. Завершение работы.")
        return

    print("📊 Оцениваем модель...")
    evaluate_model(model, X_test, y_test)

    print("🔄 Обновляем код через Gemini API...")
    update_code()

    # Генерация сообщения для коммита
    commit_msg = generate_commit_message(changes_summary="Автоматическое обновление модели, анализа и кода")

    print(f"✅ Обновление завершено. Название коммита: {commit_msg}")

if __name__ == "__main__":
    main()

