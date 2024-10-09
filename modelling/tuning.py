import logging

from sklearn.model_selection import GridSearchCV


def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5,
                         scoring='accuracy'):
    """
    Function to perform hyperparameter tuning using GridSearchCV.

    Parameters:
    - model: The machine learning model (e.g., SVC(), RandomForestClassifier(), MultinomialNB())
    - param_grid: The grid of hyperparameters for tuning
    - X_train: Training feature data
    - y_train: Training target data
    - cv: Cross-validation strategy (default=5)
    - scoring: The metric for evaluating performance (default='accuracy')

    Returns:
    - best_params: The best set of hyperparameters found
    - best_score: The best cross-validation score achieved
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv,
                               scoring=scoring, verbose=1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logging.INFO(f"Best parameters found for {model.__class__.__name__}: ",
                 best_params)
    logging.INFO(f"Best cross-validation score: ", best_score)

    return best_params, best_score