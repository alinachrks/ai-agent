import os
import datetime
from data_analysis import exploratory_data_analysis

# Папка для отчётов
REPORTS_DIR = "reports"
REPORT_FILE = os.path.join(REPORTS_DIR, "data_report.md")

def generate_report():
    """Генерирует отчёт с новыми инсайтами по данным."""
    
    # Запускаем анализ данных
    exploratory_data_analysis()

    # Читаем текущий отчёт
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            report_content = f.read()
    else:
        report_content = "❌ Ошибка: отчёт не найден."

    # Генерируем новые выводы
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insights = f"""
    ### 📈 Новые инсайты (автоанализ)
    📅 Дата обновления: {timestamp}

    - Анализ данных показал, что {len(report_content.splitlines())} строк в отчёте.
    - Корреляционные данные выявили потенциальные зависимости в датасете.
    - Требуется дополнительный анализ для выявления аномалий.
    """

    # Добавляем инсайты к отчёту
    updated_report = report_content + "\n\n" + insights

    # Записываем обновлённый отчёт
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(updated_report)

    print(f"✅ Отчёт обновлён и сохранён в {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()
