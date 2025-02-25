import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
def load_data():
    df = sns.load_dataset("iris")
    
    # Возможность заменить пропущенные значения, если они есть
    if df.isnull().sum().sum() > 0:
        df.fillna(df.median(), inplace=True)  # Заполнение пропусков медианой
        print("Пропущенные значения были заменены медианой.")
    
    return df

# Анализ данных и генерация отчёта
def exploratory_data_analysis(df):
    # Начальный отчет о данных
    report = f"""## Автоматический анализ данных
    
    **Размерность данных:** {df.shape}
    
    **Пропущенные значения:**\n{df.isnull().sum()}
    
    **Основные статистики:**\n{df.describe()}
    
    **Корреляция между признаками:**\n{df.corr()}
    """
    
    # Генерация визуализаций

    # Парный график
    pairplot = sns.pairplot(df, hue=df.columns[-1])
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
