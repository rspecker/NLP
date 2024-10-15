import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, \
    classification_report, confusion_matrix


def evaluate_model(best_model: sklearn.base.BaseEstimator,
                   x_test: pd.DataFrame | np.array,
                   y_test: pd.Series | np.array):
    """
    Evaluate the performance of a trained model on test data.

    Args:
        best_model: The trained model to evaluate.
        x_test: Test data features.
        y_test: Test data labels.

    Returns:
        test_accuracy: Accuracy of the model on the test data.
        classification_rep: Classification report with precision,
            recall, and F1-score.
        cm: Confusion matrix for the test data.
    """
    y_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return test_accuracy, classification_rep, cm


def save_results_to_file(model_name: str, best_params: dict, best_score: float,
                         test_accuracy: float, classification_rep: str):
    """
    Save grid search results and model evaluation metrics to a text file.

    Args:
        model_type: Type of model, sentence embeddings, word embeddings,
            information retrieval, TF-IDF, etc.
        model_name: Name of the model.
        best_params: Best hyperparameters from grid search.
        best_score: Best cross-validation score during grid search.
        test_accuracy: Accuracy of the model on the test data.
        classification_rep: Classification report with precision, recall,
            and F1-score.

    Returns:
        None
    """
    result_file_path = f'results/{model_name}/grid_search_results.txt'
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Model: {model_name}\n")
        result_file.write(f"Best Parameters: {best_params}\n")
        result_file.write(f"Best Cross-Validation Score: {best_score:.4f}\n")
        result_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        result_file.write(f"Classification Report:\n{classification_rep}\n")


def save_confusion_matrix(model_type: str, model_name: str, cm: np.array,
                          y_test: pd.Series | np.array):
    """
    Save the confusion matrix as an image file.

    Args:
        model_type: Type of model, sentence embeddings, word embeddings,
            information retrieval, TF-IDF, etc.
        model_name: Name of the model.
        cm: Confusion matrix from the test data.
        y_test: Test data labels.

    Returns:
        None
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(f'results/{model_type}/{model_name}/confusion_matrix.png')
    plt.close()
