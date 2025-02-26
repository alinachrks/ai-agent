import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def exploratory_data_analysis(df, report_path="report.md", output_dir="eda_outputs"):
    """
    Выполняет анализ данных (EDA), создаёт графики и записывает отчёт.
    :param df: DataFrame с признаками
    :param report_path: Путь для сохранения отчёта
    :param output_dir: Директория для сохранения графиков
    """
    print("🔍 Выполняем анализ данных...")

    # Проверяем тип данных, чтобы убедиться, что передан DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Ошибка: ожидался pandas DataFrame, а получено {type(df)}")

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
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Матрица корреляции")
    correlation_plot_path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(correlation_plot_path)
    plt.close()

    # Гистограммы всех признаков
    df.hist(figsize=(12, 8), bins=30)
    histograms_plot_path = os.path.join(output_dir, "histograms.png")
    plt.savefig(histograms_plot_path)
    plt.close()

    # Парные графики (если колонок меньше 6, иначе первые 5)
    selected_columns = df.columns[:5] if len(df.columns) > 5 else df.columns
    pairplot_fig = sns.pairplot(df[selected_columns])
    pairplot_path = os.path.join(output_dir, "pairplot.png")
    pairplot_fig.savefig(pairplot_path)  # Теперь корректный вызов
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
        f.write(f"{df.corr().to_string()}\n\n")
        f.write("### Визуализации\n")
        f.write(f"![Матрица корреляции]({correlation_plot_path})\n")
        f.write(f"![Гистограммы]({histograms_plot_path})\n")
        f.write(f"![Парные графики]({pairplot_path})\n")

    print(f"📊 Анализ данных завершён. Отчёт сохранён в {report_path}")
