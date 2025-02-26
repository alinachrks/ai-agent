import os
from data_analysis import exploratory_data_analysis
from report_generator import generate_report
from db_connector import connect_db

def main():
    """
    🔹 Основной процесс:
    1️⃣ Подключение к БД
    2️⃣ Анализ данных (EDA)
    3️⃣ Генерация отчёта
    """

    print("📥 Подключаемся к базе данных...")
    conn = connect_db()
    if conn is None:
        print("❌ Ошибка: не удалось подключиться к БД.")
        return

    # Проверяем, существует ли папка для отчётов
    if not os.path.exists("reports/"):
        print("❌ Папка reports/ отсутствует. Создаём...")
        os.makedirs("reports", exist_ok=True)
        os.makedirs("reports/images", exist_ok=True)
    else:
        print("✅ Папка reports/ уже существует.")

    print("📊 Запускаем анализ данных...")
    exploratory_data_analysis()

    print("📝 Генерируем отчёт...")
    generate_report()

    print("✅ Процесс завершён!")

if __name__ == "__main__":
    main()
