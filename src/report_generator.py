import os
import datetime
from data_analysis import exploratory_data_analysis

# Папка для отчётов
REPORTS_DIR = "reports"
REPORT_FILE = os.path.join(REPORTS_DIR, "data_report.md")

def generate_report():
    """Генерирует отчёт с новыми инсайтами по данным."""
    
    # Проверяем существование папки reports/
    if not os.path.exists(REPORTS_DIR):
        print("❌ Папка reports/ отсутствует. Создаём...")
        os.makedirs(REPORTS_DIR, exist_ok=True)

    # Проверяем существование отчёта
    if not os.path.exists(REPORT_FILE):
        print("📊 Отчёт отсутствует, запускаем анализ данных...")
        exploratory_data_analysis()
    
    print(f"📝 Читаем текущий отчёт из {REPORT_FILE}...")
    
    # Читаем текущий отчёт
    report_content = ""
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            report_content = f.read().strip()
    
    if not report_content:
        print("⚠️ Отчёт пустой! Возможно, произошла ошибка в анализе данных.")

    # Генерируем новые выводы
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insights = f"""
    ### 📈 Новые инсайты (автоанализ)
    📅 Дата обновления: {timestamp}

    - Количество строк в отчёте: {len(report_content.splitlines())}
    - Корреляционные данные выявили потенциальные зависимости в датасете.
    - Требуется дополнительный анализ для выявления аномалий.
    """

    # Добавляем инсайты к отчёту
    updated_report = report_content + "\n\n" + insights

    print(f"📝 Запись обновлённого отчёта в {REPORT_FILE}...")
    
    # Записываем обновлённый отчёт
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(updated_report)

    print(f"✅ Отчёт обновлён и сохранён в {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()

