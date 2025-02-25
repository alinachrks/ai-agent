import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def exploratory_data_analysis(df, report_path="report.md", output_dir="eda_outputs"):
    """
    Выполняет анализ данных (EDA), создаёт графики и записывает отчёт.
    :param df: DataFrame с данными
    :param report_path: Путь для сохранения отчёта
    :param output_dir: Директория для сохранения графиков
    """
    print("🔍 Выполняем анализ данных...")

    # Создаём директорию для графиков, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Проверяем пропущенные значения
    missing_values = df.isnull().sum()
    missing_summary = missing_values[missing_values > 0]
    if not missing_summary.empty:
        df.fillna(df.median(), inplace=True)  # Заполняем пропуски медианой
        print("⚠️ Пропущенные значения были заменены медианой.")

    # Описательные статистики
    stats_summary = df.describe()

    # Корреляционная матрица
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Матрица корреляции")
    correlation_plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_plot_path)
    plt.close()

    # Гистограммы всех признаков
    df.hist(figsize=(12, 8), bins=30)
    histograms_plot_path = os.path.join(output_dir, "histograms.png")
    plt.savefig(histograms_plot_path)
    plt.close()

    # Парные графики для первых 5 признаков
    selected_columns = df.columns[:5]
    sns.pairplot(df[selected_columns])
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    plt.savefig(pairplot_path)
    plt.close()

    # Запись отчёта
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("## Автоматический анализ данных\n\n")
        f.write(f"**Размерность данных:** {df.shape}\n\n")
        f.write("### Пропущенные значения\n")
        f.write(f"{missing_summary.to_string()}\n\n" if not missing_summary.empty else "Нет пропущенных значений.\n\n")
        f.write("### Основные статистики\n")
        f.write(f"{stats_summary.to_string()}\n\n")
        f.write("### Корреляция между признаками\n")
        f.write(f"{correlation_matrix.to_string()}\n\n")
        f.write("### Визуализации\n")
        f.write(f"![Матрица корреляции]({correlation_plot_path})\n")
        f.write(f"![Гистограммы]({histograms_plot_path})\n")
        f.write(f"![Парные графики]({pairplot_path})\n")

    print(f"📊 Анализ данных завершён. Отчёт сохранён в {report_path}")
= sns.pairplot(df, hue=df.columns[-1])
    pairplot.fig.savefig("eda_pairplot.png")
    
    # Тепловая карта корреляций
    corr_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Тепловая карта корреляции")
    plt.savefig("eda_correlation_heatmap.png")
    
    # Гистограммы распределений
    df.hist(figsize=(10, 8), bins=20, grid=False)
    plt.suptitle("Гистограммы распределений признаков")
    plt.savefig("eda_histograms.png")

    # График распределения для каждого признака
    for column in df.columns[:-1]:  # Исключаем целевую переменную
        plt.figure(figsize=(6, 4))
        sns.histplot(df[column], kde=True)
        plt.title(f"Распределение {column}")
        plt.savefig(f"eda_{column}_distribution.png")

    return report
