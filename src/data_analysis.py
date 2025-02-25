import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
def load_data():
    df = sns.load_dataset("iris")
    return df

# Анализ данных и генерация отчёта
def exploratory_data_analysis(df):
    report = f"""## Автоматический анализ данных
    
    **Размерность данных:** {df.shape}
    
    **Пропущенные значения:**\n{df.isnull().sum()}
    
    **Основные статистики:**\n{df.describe()}
    """

    # Генерация визуализаций
    sns.pairplot(df, hue=df.columns[-1])
    plt.savefig("eda_plot.png")

    return report
