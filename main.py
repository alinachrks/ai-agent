import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Загрузка данных (пример: встроенный датасет Iris)
def load_data():
    df = sns.load_dataset("iris")
    return df

# Автоматический анализ данных
def exploratory_data_analysis(df):
    report = """## Автоматический анализ данных
    
    **Размерность данных:** {shape}
    
    **Пропущенные значения:**
    {missing}
    
    **Основные статистики:**
    {desc}
    
    **Корреляции:**
    
    """.format(
        shape=df.shape,
        missing=df.isnull().sum(),
        desc=df.describe()
    )
    
    # Генерация визуализаций
    sns.pairplot(df, hue=df.columns[-1])
    plt.savefig("eda_plot.png")
    
    return report

# Определение типа задачи и обучение модели
def train_model(df):
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if y.dtype == 'O':  # Классификация
        model = RandomForestClassifier()
    else:  # Регрессия
        model = RandomForestRegressor()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if y.dtype == 'O':
        metric = accuracy_score(y_test, y_pred)
        metric_name = "Accuracy"
    else:
        metric = mean_squared_error(y_test, y_pred, squared=False)
        metric_name = "RMSE"
    
    return model, metric_name, metric

if __name__ == "__main__":
    df = load_data()
    eda_report = exploratory_data_analysis(df)
    model, metric_name, metric = train_model(df)
    
    with open("report.md", "w") as f:
        f.write(eda_report)
        f.write(f"\n\n**Метрика модели ({metric_name}): {metric:.4f}**")
    
    print("Готово! Анализ и обучение завершены.")
