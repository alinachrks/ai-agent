from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Обучение модели
def train_model(df):
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y.dtype == 'O':  # Классификация
        model = RandomForestClassifier()
        metric_func = accuracy_score
        metric_name = "Accuracy"
    else:  # Регрессия
        model = RandomForestRegressor()
        metric_func = mean_squared_error
        metric_name = "RMSE"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metric = metric_func(y_test, y_pred)

    return model, metric_name, metric
