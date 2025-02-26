This improved code incorporates several enhancements for modularity, flexibility, and broader applicability:

```python
import pandas as pd
import openml
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.exceptions import NotFittedError
import time
import os
import json
import logging
import joblib
from typing import Tuple, Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

def load_data(dataset_id: int) -> Union[pd.DataFrame, None]:
    """Loads data from OpenML."""
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        df, _, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        logging.info(f"Loaded dataset {dataset_id} from OpenML.")
        return df
    except Exception as e:
        logging.exception(f"Error loading data: {e}")
        return None

def preprocess_data(df: pd.DataFrame, target_column: str = "Class", 
                    imputation_strategy: str = "mean", encoding: str = "one-hot") -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocesses the data."""
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        #Handle Missing Values
        numerical_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        imputer_num = SimpleImputer(strategy=imputation_strategy) if imputation_strategy in ["mean", "median", "most_frequent"] else KNNImputer(n_neighbors=5) if imputation_strategy == "knn" else None
        if imputer_num:
            X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

        #Handle Categorical Features
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) if encoding == "one-hot" else LabelEncoder() if encoding == "label" else None
        if encoder:
            if encoding == "one-hot":
                encoded_data = encoder.fit_transform(X[categorical_cols])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
                X = pd.concat([X[numerical_cols], encoded_df], axis=1)
            else:  # label encoding
                for col in categorical_cols:
                    X[col] = encoder.fit_transform(X[col])
        X = X.fillna(0) #Fill any remaining NaN after OneHotEncoding/Label Encoding

        #Scale Numerical Features
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        return X, y
    except Exception as e:
        logging.exception(f"Error preprocessing data: {e}")
        return None, None

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model: Any, param_grid: Dict[str, Any] = None) -> Any:
    """Trains a model."""
    try:
        start_time = time.time()
        if param_grid:
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'precision': make_scorer(precision_score, average='weighted'),
                       'recall': make_scorer(recall_score, average='weighted'),
                       'f1': make_scorer(f1_score, average='weighted')}
            grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='f1', cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            logging.info(f"Best hyperparameters: {grid_search.best_params_}")
            logging.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)
        elapsed_time = time.time() - start_time
        logging.info(f"Training completed in {elapsed_time:.2f} seconds.")
        return model
    except NotFittedError as e:
        logging.error(f"Model not fitted: {e}")
        return None
    except Exception as e:
        logging.exception(f"Error training model: {e}")
        return None

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, Any, Any]:
    """Evaluates a trained model."""
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return report, y_pred, cm
    except NotFittedError as e:
        logging.error(f"Model not fitted: {e}")
        return None, None, None
    except Exception as e:
        logging.exception(f"Error evaluating model: {e}")
        return None, None, None

def save_results(results: Dict[str, Any], model: Any, model_name: str, path: str) -> None:
    """Saves results and model to file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        joblib.dump(model, path.replace(".json", ".pkl"))
        logging.info(f"Results and model saved to {path.replace('.json','.pkl')}")
    except Exception as e:
        logging.exception(f"Error saving results: {e}")

def run_experiment(dataset_id: int, models: Dict[str, Tuple[Any, Dict[str, Any]]], 
                   imputation_strategy: str = "mean", encoding: str = "one-hot", test_size: float = 0.3, random_state: int = 42) -> None:
    """Runs a complete machine learning experiment."""
    df = load_data(dataset_id)
    if df is None:
        return

    X, y = preprocess_data(df, imputation_strategy=imputation_strategy, encoding=encoding)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    for model_name, (model, param_grid) in models.items():
        trained_model = train_model(X_train, y_train, model, param_grid)
        if trained_model is not None:
            report, y_pred, cm = evaluate_model(trained_model, X_test, y_test)
            if report is not None:
                results = {
                    "model": model_name,
                    "params": param_grid if param_grid else {},
                    "dataset_id": dataset_id,
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                }
                save_results(results, trained_model, model_name, path=f"results/{dataset_id}_{model_name}.json")

if __name__ == "__main__":
    dataset_id = 1464
    imputation_strategy = "knn"
    encoding = "one-hot"
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5]
            }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5]
            }),
        "VotingClassifier": (VotingClassifier(estimators=[("rf", RandomForestClassifier(random_state=42)), ("gb", GradientBoostingClassifier(random_state=42))], voting='soft'), {}),

    }

    run_experiment(dataset_id, models, imputation_strategy, encoding)

```

Key improvements in this version:

* **Type hints:** Added type hints for better readability and maintainability.
* **Modular design:** The code is broken down into smaller, more focused functions, improving readability and reusability.
* **Improved imputation:** Added `median` and `most_frequent` strategies to `SimpleImputer`.
* **Added VotingClassifier:** Demonstrates the ease of adding new models. The `run_experiment` function handles both models with and without hyperparameter tuning seamlessly.
* **More informative logging:** More specific messages are logged during the experiment's execution.
* **Flexible parameterization:** The `run_experiment` function accepts parameters like `test_size` and `random_state`, allowing for easy experimentation.
* **Error handling:**  More comprehensive error handling in all functions.


This revised code is significantly more organized, easier to extend, and better suited for managing larger or more complex machine learning projects.  Remember to `pip install joblib`.
