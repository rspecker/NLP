import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV


def perform_grid_search(model: sklearn.base.BaseEstimator, param_grid: dict,
                        model_type: str, model_name: str,
                        x_train: pd.DataFrame | np.array,
                        y_train: pd.Series | np.array):
    """
    Perform grid search to find the best hyperparameters for a given model.

    Args:
        model: The machine learning model to optimize.
        param_grid: Dictionary with parameters names (str) as keys and lists
            of parameter settings to try as values.
        model_type: Type of model, sentence embeddings, word embeddings,
            information retrieval, TF-IDF, etc.
        model_name: Name of the model to use for saving the best model.
        x_train: Training data features.
        y_train: Training data labels.

    Returns:
        best_model: The model with the best hyperparameters.
        best_params: Best hyperparameters found during grid search.
        best_score: Best cross-validation score during grid search.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                               n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, f'results/{model_type}/{model_name}_best_model.pkl')

    return best_model, grid_search.best_params_, grid_search.best_score_
