This improved code addresses several areas for enhancement:  handling imbalanced datasets, adding more sophisticated evaluation metrics, improving the logging,  and enhancing model selection capabilities.


```python
import pandas as pd
import openml
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

# ... (load_data and preprocess_data functions remain largely unchanged) ...

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model: Any, param_grid: Dict[str, Any] = None, class_weights: Dict[str, float] = None) -> Any:
    """Trains a model, handling class weights."""
    try:
        start_time = time.time()
        if param_grid:
            # Use RandomizedSearchCV for faster hyperparameter tuning with larger parameter spaces
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'balanced_accuracy': make_scorer(balanced_accuracy_score),
                       'precision': make_scorer(precision_score, average='weighted'),
                       'recall': make_scorer(recall_score, average='weighted'),
                       'f1': make_scorer(f1_score, average='weighted'),
                       'roc_auc': make_scorer(roc_auc_score, average='weighted', multi_class='ovr')} # added roc_auc

            if isinstance(param_grid, dict): #Check if param_grid is a dictionary or a list of dictionaries.
              grid_search = RandomizedSearchCV(model, param_grid, scoring=scoring, refit='f1', cv=5, n_jobs=-1, verbose=1, n_iter=10, random_state=42) # added n_iter and random_state
            else:
              grid_search = RandomizedSearchCV(model, param_grid, scoring=scoring, refit='f1', cv=5, n_jobs=-1, verbose=1, random_state=42) # added random_state
            grid_search.fit(X_train, y_train, sample_weight = class_weights) # added sample_weight
            model = grid_search.best_estimator_
            logging.info(f"Best hyperparameters: {grid_search.best_params_}")
            logging.info(f"Best F1 score: {grid_search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train, sample_weight = class_weights) # added sample_weight

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
    """Evaluates a trained model with additional metrics."""
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) #Added probability predictions for AUC
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


def save_results(results: Dict[str, Any], model: Any, model_name: str, path: str) -> None:
    # ... (function remains largely unchanged) ...


def run_experiment(dataset_id: int, models: Dict[str, Tuple[Any, Dict[str, Any]]],
                   imputation_strategy: str = "mean", encoding: str = "one-hot", test_size: float = 0.3, random_state: int = 42, handle_imbalance: str = None) -> None:
    """Runs a complete machine learning experiment, handling class imbalance."""
    df = load_data(dataset_id)
    if df is None:
        return

    X, y = preprocess_data(df, imputation_strategy=imputation_strategy, encoding=encoding)
    if X is None or y is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    #Handle class imbalance
    if handle_imbalance == "class_weight":
        class_weights = class_weight.compute_sample_weight('balanced', y_train)
    elif handle_imbalance == "oversample":
      # Add oversampling techniques here if needed (e.g., SMOTE)
      pass  
    else:
        class_weights = None


    for model_name, (model, param_grid) in models.items():
        trained_model = train_model(X_train, y_train, model, param_grid, class_weights=class_weights)
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
                    "balanced_accuracy": balanced_acc, # Added balanced accuracy
                    "precision": precision_score(y_test, y_pred, average='weighted'),
                    "recall": recall_score(y_test, y_pred, average='weighted'),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "roc_auc": auc # Added AUC
                }
                save_results(results, trained_model, model_name, path=f"results/{dataset_id}_{model_name}.json")



if __name__ == "__main__":
    dataset_id = 1464
    imputation_strategy = "knn"
    encoding = "one-hot"
    handle_imbalance = "class_weight" # Added to handle class imbalance
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': randint(100, 201), # changed to randint for RandomizedSearchCV
            'max_depth': [None] + list(randint(5, 51).rvs(5)), #added list of random max_depth values
            'min_samples_split': randint(2,11) #changed to randint for RandomizedSearchCV
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            'n_estimators': randint(100, 201), # changed to randint for RandomizedSearchCV
            'learning_rate': uniform(0.01, 0.1), #changed to uniform for RandomizedSearchCV
            'max_depth': randint(3,6) #changed to randint for RandomizedSearchCV
        }),
        "VotingClassifier": (VotingClassifier(estimators=[("rf", RandomForestClassifier(random_state=42)), ("gb", GradientBoostingClassifier(random_state=42))], voting='soft'), {}),
    }

    run_experiment(dataset_id, models, imputation_strategy, encoding, handle_imbalance=handle_imbalance)
```

Key improvements:

* **Handling Class Imbalance:** The `run_experiment` function now includes an `handle_imbalance` parameter.  Currently, it supports `"class_weight"` which uses `class_weight.compute_sample_weight` to balance the classes during training.  You could easily extend this to include oversampling techniques (like SMOTE)  by adding a corresponding `elif` block.
* **Improved Hyperparameter Tuning:** Using `RandomizedSearchCV` instead of `GridSearchCV` for faster exploration of a wider parameter space, especially helpful with large search spaces.  Distributions are now used for parameter selection, allowing a more efficient search
* **Additional Evaluation Metrics:** The `evaluate_model` function now returns balanced accuracy and AUC (Area Under the ROC Curve), providing a more comprehensive evaluation, especially useful when dealing with imbalanced datasets.  The `roc_auc` score is calculated using `multi_class='ovr'` (One-vs-Rest) for multi-class problems and for binary problems the AUC for the positive class is computed.
* **More Robust Logging:** The logging is improved to provide more detail about the experiment's progress.
* **Clearer Parameter Handling**: The code is organized to make it easier to add or modify models and parameters.


Remember to install `scikit-learn` and other necessary libraries.  This enhanced version offers greater flexibility, robustness, and more informative results.  Remember that adding oversampling may require installing `imblearn`.
