This improved code incorporates several optimizations and additions, focusing on modularity, robustness, and extensibility.

```python
import pandas as pd
import openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.exceptions import NotFittedError
from sklearn.utils import class_weight
import time
import os
import json
import logging
import joblib
from typing import Tuple, Dict, Any, Union
import numpy as np
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE # Added SMOTE for oversampling


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

# --- Data Loading and Preprocessing ---

def load_data(dataset_id: int) -> Union[pd.DataFrame, None]:
    """Loads data from OpenML."""
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        df = pd.DataFrame(X, columns=attribute_names)
        df[dataset.default_target_attribute] = y
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def preprocess_data(df: pd.DataFrame, imputation_strategy: str = "mean", encoding: str = "one-hot") -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocesses the data."""
    try:
        X = df.drop(columns=df.columns[-1])
        y = df.iloc[:, -1]

        #Handle Missing Values
        imputer = SimpleImputer(strategy=imputation_strategy) if imputation_strategy != 'knn' else KNNImputer()
        X = imputer.fit_transform(X)
        X = pd.DataFrame(X, columns=df.columns[:-1])


        #Handle Categorical features
        if encoding == "one-hot":
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X = encoder.fit_transform(X)
            feature_names = list(encoder.get_feature_names_out(df.columns[:-1]))
            X = pd.DataFrame(X, columns = feature_names)
        elif encoding == "label":
          for col in X.columns:
            if X[col].dtype == 'object':
              le = LabelEncoder()
              X[col] = le.fit_transform(X[col])


        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None


# --- Model Training and Evaluation ---

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model: Any, param_grid: Dict[str, Any] = None, class_weights: np.ndarray = None, oversample: bool = False) -> Any:
    """Trains a model, handling class weights and oversampling."""
    try:
        start_time = time.time()
        if oversample:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        if param_grid:
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'balanced_accuracy': make_scorer(balanced_accuracy_score),
                       'precision': make_scorer(precision_score, average='weighted'),
                       'recall': make_scorer(recall_score, average='weighted'),
                       'f1': make_scorer(f1_score, average='weighted'),
                       'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr')}

            grid_search = RandomizedSearchCV(model, param_grid, scoring=scoring, refit='f1', cv=5, n_jobs=-1, verbose=1, n_iter=10, random_state=42)
            grid_search.fit(X_train, y_train, sample_weight=class_weights)
            model = grid_search.best_estimator_
            logging.info(f"Best hyperparameters: {grid_search.best_params_}")
            logging.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train, sample_weight=class_weights)

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
        y_prob = model.predict_proba(X_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if len(np.unique(y_test)) > 2 else roc_auc_score(y_test, y_prob[:, 1])
        return report, y_pred, cm, balanced_acc, auc
    except NotFittedError as e:
        logging.error(f"Model not fitted: {e}")
        return None, None, None, None, None
    except Exception as e:
        logging.exception(f"Error evaluating model: {e}")
        return None, None, None, None, None

# --- Results Handling ---

def save_results(results: Dict[str, Any], model: Any, model_name: str, path: str) -> None:
    """Saves results to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    joblib.dump(model, f"{path}.pkl") #Saves the model as well


# --- Experiment Runner ---

def run_experiment(dataset_id: int, models: Dict[str, Tuple[Any, Dict[str, Any]]],
                   imputation_strategy: str = "mean", encoding: str = "one-hot", test_size: float = 0.3, random_state: int = 42, 
                   handle_imbalance: str = None) -> None:
    """Runs a complete machine learning experiment."""
    df = load_data(dataset_id)
    if df is None:
        return

    X, y = preprocess_data(df, imputation_strategy, encoding)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    class_weights = None
    oversample = False
    if handle_imbalance == "class_weight":
        class_weights = class_weight.compute_sample_weight('balanced', y_train)
    elif handle_imbalance == "oversample":
        oversample = True

    for model_name, (model, param_grid) in models.items():
        trained_model = train_model(X_train, y_train, model, param_grid, class_weights, oversample)
        if trained_model is not None:
            report, y_pred, cm, balanced_acc, auc = evaluate_model(trained_model, X_test, y_test)
            if report is not None:
                results = {
                    "model": model_name,
                    "params": param_grid if param_grid else {},
                    "dataset_id": dataset_id,
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_acc,
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "roc_auc": auc
                }
                save_results(results, trained_model, model_name, path=f"results/{dataset_id}_{model_name}.json")

if __name__ == "__main__":
    dataset_id = 1464
    imputation_strategy = "knn"
    encoding = "one-hot"
    handle_imbalance = "oversample" # changed to oversample to test SMOTE
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': randint(100, 201),
            'max_depth': [None] + list(randint(5, 51).rvs(5)),
            'min_samples_split': randint(2, 11)
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            'n_estimators': randint(100, 201),
            'learning_rate': uniform(0.01, 0.1),
            'max_depth': randint(3, 6)
        }),
        "VotingClassifier": (VotingClassifier(estimators=[("rf", RandomForestClassifier(random_state=42)), ("gb", GradientBoostingClassifier(random_state=42))], voting='soft'), {}),
    }

    run_experiment(dataset_id, models, imputation_strategy, encoding, handle_imbalance=handle_imbalance)
```

Key improvements in this version:

* **Modularity:** The code is broken down into more focused functions (e.g., `load_data`, `preprocess_data`, `train_model`, `evaluate_model`, `save_results`). This improves readability, maintainability, and testability.
* **Improved Preprocessing:** The `preprocess_data` function now handles both missing values (using either `SimpleImputer` or `KNNImputer`) and categorical features (using `OneHotEncoder` or `LabelEncoder`).  The choice of imputation and encoding strategy is now passed as parameters.
* **Oversampling with SMOTE:** Added support for SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets.  This is activated using  `handle_imbalance = "oversample"`.  Remember to install `imblearn`: `pip install imbalanced-learn`
* **Model Saving:** The `save_results` function now also saves the trained model using `joblib.dump`. This allows you to reload and reuse the trained model without retraining.
* **Error Handling:** More robust error handling is included throughout the code to catch and log potential issues.
* **Type Hints:**  Added type hints for better code readability and maintainability.


This refined version is more organized, efficient, and adaptable to various datasets and model configurations.  The modular design makes it easier to experiment with different preprocessing techniques, models, and hyperparameter tuning strategies. Remember to install `imbalanced-learn` if you want to use the SMOTE oversampling.
