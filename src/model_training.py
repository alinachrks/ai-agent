This improved code adds several features, including more robust error handling,  better preprocessing options,  experiment tracking, and the ability to easily switch datasets and models.

```python
import pandas as pd
import openml
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
import time
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset_id=1464):
    """Loads data from OpenML."""
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        df, _, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        logging.info(f"Loaded dataset {dataset_id} from OpenML.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def preprocess_data(df, target_column="Class", imputation_strategy="mean", encoding="one-hot"):
    """Preprocesses the data: handles missing values, encodes categorical features, and scales numerical features."""
    try:
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        #Handle Missing Values
        if imputation_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif imputation_strategy == "knn":
            imputer = KNNImputer(n_neighbors=5) # Adjust n_neighbors as needed
        else:
            raise ValueError("Invalid imputation strategy. Choose 'mean' or 'knn'.")
        X = imputer.fit_transform(X)


        #Identify categorical and numerical features (after imputation to avoid errors)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numerical_cols = df.select_dtypes(include=['number']).columns


        if encoding == "one-hot":
          encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) #handle_unknown for unseen categories in test
          categorical_data = encoder.fit_transform(df[categorical_cols])
          X = pd.concat([pd.DataFrame(X, columns=numerical_cols), pd.DataFrame(categorical_data)], axis=1)
          
        elif encoding == "label":
            label_encoder = LabelEncoder()
            for col in categorical_cols:
                X[col] = label_encoder.fit_transform(X[col])
        else:
            raise ValueError("Invalid encoding strategy. Choose 'one-hot' or 'label'.")

        #Scale Numerical Features
        scaler = StandardScaler()
        numerical_data = scaler.fit_transform(X[numerical_cols])
        X = pd.DataFrame(numerical_data, columns=numerical_cols)

        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid=None):
    """Trains and evaluates a model, optionally using GridSearchCV for hyperparameter tuning."""
    try:
        if param_grid:
            scoring = {'accuracy': make_scorer(accuracy_score),
                       'precision': make_scorer(precision_score, average='weighted'),
                       'recall': make_scorer(recall_score, average='weighted'),
                       'f1': make_scorer(f1_score, average='weighted')}
            grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='f1', cv=5, n_jobs=-1, verbose=1)
            start_time = time.time()
            grid_search.fit(X_train, y_train)
            elapsed_time = time.time() - start_time
            logging.info(f"GridSearchCV completed in {elapsed_time:.2f} seconds.")
            best_model = grid_search.best_estimator_
            logging.info("Best hyperparameters: %s", grid_search.best_params_)
            logging.info("Best F1 score: %.4f", grid_search.best_score_)
            y_pred = best_model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return report, y_pred, cm
    except Exception as e:
        logging.error(f"Error training/evaluating model: {e}")
        return None, None, None

def save_results(report, y_pred, y_test, cm, model_name, params, dataset_id, path="results.json"):
    """Saves the results to a JSON file."""
    results = {
        "model": model_name,
        "params": params,
        "dataset_id": dataset_id,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
    }
    try:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results saved to {path}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")


if __name__ == "__main__":
    dataset_id = 1464
    imputation_strategy = "knn"  # Choose between "mean" and "knn"
    encoding = "one-hot" # Choose between "one-hot" and "label"
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
            })
    }


    df = load_data(dataset_id)
    if df is not None:
        for model_name, (model, param_grid) in models.items():
          X, y = preprocess_data(df, imputation_strategy=imputation_strategy, encoding=encoding)
          if X is not None and y is not None:
              X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
              report, y_pred, cm = train_and_evaluate_model(X_train, y_train, X_test, y_test, model, param_grid=param_grid)
              if report is not None:
                  save_results(report, y_pred, y_test, cm, model_name, param_grid, dataset_id, path=f"results/{dataset_id}_{model_name}.json")

```

This enhanced version offers:

* **More sophisticated imputation:**  Uses `KNNImputer` as an alternative to simple mean imputation.
* **One-hot encoding or label encoding:**  Allows flexible encoding of categorical features.
* **Multiple model support:** Easily switch between different models by modifying the `models` dictionary.
* **Experiment Tracking:** Saves results (including the classification report, confusion matrix, and hyperparameters) to a JSON file for easy analysis and comparison.
* **Improved logging:**  Uses the `logging` module for better error reporting and tracking of the process.
* **Directory creation:** Creates the 'results' directory if it doesn't exist.
* **Handle unknown categories:** Uses `handle_unknown='ignore'` in OneHotEncoder to prevent errors with unseen categories during testing.

Remember to adjust the `imputation_strategy` and `encoding` parameters and the hyperparameter grids as needed for your specific dataset and model.  The `models` dictionary can be expanded to include other algorithms.  The output is now stored in a more organized way in a JSON file.  This makes it easier to compare results across different datasets and models.
