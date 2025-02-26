import datetime
from db_reader import get_table_info

TABLE_NAME = "insights"  # Название таблицы в БД

def generate_report():
    """Генерирует отчёт о состоянии базы данных."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    row_count, columns = get_table_info(TABLE_NAME)

    if isinstance(row_count, str):  # Если ошибка
        report_content = f"❌ Ошибка при работе с базой: {row_count}"
    else:
        column_info = "\n".join([f"- {col[0]} ({col[1]})" for col in columns])
        report_content = f"""
        ## 📊 Автоматический отчёт
        📅 Время обновления: {timestamp}
        
        - 🔢 Количество записей: {row_count}
        - 📄 Структура таблицы `{TABLE_NAME}`:
        
        {column_info}
        """

    with open("reports/db_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("✅ Отчёт сгенерирован и сохранён в reports/db_report.md")
