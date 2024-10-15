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
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Define hyperparameters for tuning for MultinomialNB
mnb_param_grid = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
}

# Define the models
models = {
    'SVC': (SVC(), svc_param_grid),
    'RandomForest': (RandomForestClassifier(), rf_param_grid),
    'MultinomialNB': (MultinomialNB(), mnb_param_grid)
}
