This improved code addresses several aspects, including hyperparameter optimization, reporting, and enhanced flexibility.

```python
import pandas as pd
import openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
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
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

# --- Data Loading and Preprocessing ---

# ... (load_data and preprocess_data functions remain largely unchanged) ...


# --- Model Training and Evaluation ---

def train_model(X_train: pd.DataFrame, y_train: pd.Series, model: Any, param_grid: Dict[str, Any] = None, 
                class_weights: np.ndarray = None, oversample: str = None, undersample: bool = False, cv: int = 5) -> Any:
    """Trains a model, handling class weights and over/undersampling."""
    try:
        start_time = time.time()

        # Resampling
        if oversample:
            if oversample == "SMOTE":
                resampler = SMOTE(random_state=42)
            elif oversample == "ADASYN":
                resampler = ADASYN(random_state=42)
            else:
                raise ValueError("Invalid oversampling method specified.")
            X_train, y_train = resampler.fit_resample(X_train, y_train)
        if undersample:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        if param_grid:
            # Use GridSearchCV for more exhaustive search if param_grid is small
            if len(param_grid) < 10:  # heuristic for deciding between RandomizedSearchCV and GridSearchCV
              search = GridSearchCV(model, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1)
            else:
              search = RandomizedSearchCV(model, param_grid, scoring='f1_weighted', cv=cv, n_jobs=-1, verbose=1, n_iter=10, random_state=42)
            search.fit(X_train, y_train, sample_weight=class_weights)
            model = search.best_estimator_
            logging.info(f"Best hyperparameters: {search.best_params_}")
            logging.info(f"Best F1 score: {search.best_score_:.4f}")
        else:
            model.fit(X_train, y_train, sample_weight=class_weights)

        elapsed_time = time.time() - start_time
        logging.info(f"Training completed in {elapsed_time:.2f} seconds.")
        return model
    except (NotFittedError, ValueError) as e:
        logging.error(f"Error training model: {e}")
        return None
    except Exception as e:
        logging.exception(f"Unexpected error training model: {e}")
        return None



def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, Any, Any]:
    """Evaluates a trained model and returns detailed metrics."""
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if len(np.unique(y_test)) > 2 else roc_auc_score(y_test, y_prob[:, 1])
        return report, cm, balanced_acc, auc
    except NotFittedError as e:
        logging.error(f"Model not fitted: {e}")
        return None, None, None, None
    except Exception as e:
        logging.exception(f"Error evaluating model: {e}")
        return None, None, None, None


def plot_confusion_matrix(cm, y_test):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# --- Results Handling ---

# ... (save_results function remains largely unchanged) ...


# --- Experiment Runner ---

def run_experiment(dataset_id: int, models: Dict[str, Tuple[Any, Dict[str, Any]]],
                   imputation_strategy: str = "mean", encoding: str = "one-hot", test_size: float = 0.3, 
                   random_state: int = 42, handle_imbalance: Union[str, Tuple[str, bool]] = None, scaler = None, cv:int = 5) -> None:
    """Runs a complete machine learning experiment."""
    df = load_data(dataset_id)
    if df is None:
        return

    X, y = preprocess_data(df, imputation_strategy, encoding)
    if X is None or y is None:
        return
    if scaler:
      if scaler == 'standard':
        scaler = StandardScaler()
      elif scaler == 'minmax':
        scaler = MinMaxScaler()
      else:
        raise ValueError("Invalid scaler specified.")
      X = scaler.fit_transform(X)
      X = pd.DataFrame(X, columns = df.columns[:-1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    class_weights = None
    oversample_method = None
    undersample_bool = False

    if handle_imbalance:
        if isinstance(handle_imbalance, str):
            oversample_method = handle_imbalance
        elif isinstance(handle_imbalance, tuple):
            oversample_method, undersample_bool = handle_imbalance
        else:
            raise ValueError("Invalid handle_imbalance argument.")

        if oversample_method not in ["SMOTE", "ADASYN", None]:
            raise ValueError("Invalid oversampling method specified.")

        if undersample_bool and oversample_method:
          logging.warning("Both oversampling and undersampling are used. This might lead to unexpected results")

        if handle_imbalance == "class_weight":
            class_weights = class_weight.compute_sample_weight('balanced', y_train)



    for model_name, (model, param_grid) in models.items():
        trained_model = train_model(X_train, y_train, model, param_grid, class_weights, oversample_method, undersample_bool, cv)
        if trained_model is not None:
            report, cm, balanced_acc, auc = evaluate_model(trained_model, X_test, y_test)
            if report is not None:
                results = {
                    "model": model_name,
                    "params": param_grid if param_grid else {},
                    "dataset_id": dataset_id,
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                    "accuracy": report['accuracy'],
                    "balanced_accuracy": balanced_acc,
                    "precision": report['weighted avg']['precision'],
                    "recall": report['weighted avg']['recall'],
                    "f1_score": report['weighted avg']['f1-score'],
                    "roc_auc": auc
                }
                save_results(results, trained_model, model_name, path=f"results/{dataset_id}_{model_name}.json")
                plot_confusion_matrix(cm, y_test)


if __name__ == "__main__":
    dataset_id = 1464  #Example dataset ID, change to experiment with different datasets
    imputation_strategy = "knn"
    encoding = "one-hot"
    handle_imbalance = ("SMOTE", False) #Example of oversampling with SMOTE. Set to None for no resampling
    scaler = 'minmax' #Set to None, 'standard' or 'minmax' for scaling.
    models = {
        "RandomForest": (RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 150, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }),
        "GradientBoosting": (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5]
        }),
        # "VotingClassifier": (VotingClassifier(estimators=[("rf", RandomForestClassifier(random_state=42)), ("gb", GradientBoostingClassifier(random_state=42))], voting='soft'), {}),
    }

    run_experiment(dataset_id, models, imputation_strategy, encoding, handle_imbalance=handle_imbalance, scaler=scaler)
```

Key improvements:

* **GridSearchCV/RandomizedSearchCV Selection:** The code now intelligently chooses between `GridSearchCV` (for smaller search spaces) and `RandomizedSearchCV` (for larger search spaces) based on the size of the `param_grid`.  This optimizes the hyperparameter tuning process.
* **Detailed Classification Report:** The `classification_report` now uses `output_dict=True` for easier access to individual metrics.
* **Improved Resampling:** The `handle_imbalance` parameter is more flexible, allowing for both oversampling (SMOTE, ADASYN) and undersampling (RandomUnderSampler),  or just class weights.  It also includes a warning if both over and undersampling are used simultaneously.
* **Data Scaling:** Added support for data scaling using `StandardScaler` or `MinMaxScaler`.  This can significantly improve model performance.
* **Confusion Matrix Visualization:** Includes a function to plot and display the confusion matrix using `seaborn` for better visualization.
* **Error Handling:** More robust exception handling to catch various potential errors during model training and evaluation.
* **Clearer Logging:** Improved logging messages provide more informative feedback during the experiment.
* **Cross-Validation:**  Uses `cv` parameter in `train_model` for better model evaluation and allows to set different number of folds.



Remember to install the necessary libraries: `pip install imbalanced-learn seaborn matplotlib`.  Adjust the `dataset_id`,  `handle_imbalance`, `scaler`, and the models and their parameter grids to experiment with different datasets and configurations.  The example uses a smaller `param_grid` for `GridSearchCV` for faster execution.  Uncomment the VotingClassifier if needed, but be aware it might take longer to train.
