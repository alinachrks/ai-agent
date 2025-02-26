import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from db_connector import connect_db

# Название таблицы в базе данных PostgreSQL
TABLE_NAME = "edc_population_income_geom_17"

# Папки для отчётов и графиков
REPORTS_DIR = "reports"
IMAGES_DIR = "reports/images"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def load_data():
    """Загружает данные из PostgreSQL в DataFrame."""
    conn = connect_db()
    if conn is None:
        print("❌ Ошибка: не удалось подключиться к базе.")
        return None

    try:
        df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None

def exploratory_data_analysis():
    """Выполняет анализ данных и сохраняет отчёт."""
    df = load_data()
    if df is None:
        return
    
    report_path = os.path.join(REPORTS_DIR, "data_report.md")
    
    # Основные характеристики данных
    report_content = f"""
    ## 📊 Автоматический анализ данных
    📅 Время обновления: {pd.Timestamp.now()}

    - 🔢 Количество строк: {df.shape[0]}
    - 📄 Количество колонок: {df.shape[1]}

    ### 🏷 Структура таблицы `{TABLE_NAME}`
    {df.dtypes.to_frame(name="Тип данных").to_markdown()}

    ### 📌 Основные статистики
    {df.describe().to_markdown()}
    """
    
    # Сохранение отчёта
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✅ Отчёт сохранён в {report_path}")

    # Генерация графиков
    plot_file = os.path.join(IMAGES_DIR, "eda_plot.png")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Корреляция признаков в {TABLE_NAME}")
    plt.savefig(plot_file)
    print(f"✅ График сохранён в {plot_file}")

if __name__ == "__main__":
    exploratory_data_analysis()
