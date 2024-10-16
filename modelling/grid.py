from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Define hyperparameters for tuning for SVC
svc_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced']
}

# Define hyperparameters for tuning for RandomForestClassifier
rf_param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt']
}

# Define hyperparameters for tuning for MultinomialNB
mnb_param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
}

# Define the models
classifiers = {
    'SVC': (SVC(), svc_param_grid),
    'RandomForest': (RandomForestClassifier(), rf_param_grid),
    'MultinomialNB': (MultinomialNB(), mnb_param_grid)
}
