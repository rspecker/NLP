import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, \
    classification_report, confusion_matrix


def evaluate_model(best_model: sklearn.base.BaseEstimator,
                   x_test: pd.DataFrame | np.ndarray,
                   y_test: pd.Series | np.ndarray):
    """
    Evaluate the performance of a trained model on test data.

    Args:
        best_model: The trained model to evaluate.
        x_test: Test data features.
        y_test: Test data labels.

    Returns:
        y_pred: Predictions of the model on the test data.
        test_accuracy: Accuracy of the model on the test data.
        classification_rep: Classification report with precision,
            recall, and F1-score.
        cm: Confusion matrix for the test data.
    """
    y_pred = best_model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, test_accuracy, classification_rep, cm


def apply_model_to_unlabeled_data(best_model: sklearn.base.BaseEstimator,
                                  df_unlabeled: pd.DataFrame,
                                  model_type: str, model_name: str):
    """
    Apply the best model to the unlabeled test data and save the predictions
    to a text file.

    Args:
        best_model: The trained model to apply to the test data.

    Returns:
        None
    """
    pred = best_model.predict(df_unlabeled)
    # store the predictions in a results.txt file
    with open(f'results/{model_type}/{model_name}/results.txt', 'w') as f:
        f.write("genre\n")
        for i in range(len(pred)):
            f.write(f"{pred[i]}\n")


def save_results_to_file(model_type: str, model_name: str, best_params: dict,
                         best_score: float,
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
    result_file_path = f'results/{model_type}/{model_name}/grid_search_results.txt'
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Model: {model_name}\n")
        result_file.write(f"Best Parameters: {best_params}\n")
        result_file.write(f"Best Cross-Validation Score: {best_score:.4f}\n")
        result_file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        result_file.write(f"Classification Report:\n{classification_rep}\n")


def save_test_data_with_predictions(X_test, y_test, y_pred, model_type, model_name):
    """
    Save the test data along with the actual and predicted labels to a CSV file.

    Args:
        X_test (pd.DataFrame or np.array): Test data features.
        y_test (pd.Series or np.array): Actual labels for the test data.
        y_pred (np.array): Predicted labels by the model.
        model_name (str): Name of the model to use for saving the file.

    Returns:
        None
    """
    # Add the actual and predicted labels as columns
    test_data_with_preds = X_test.copy()
    test_data_with_preds['Actual'] = y_test
    test_data_with_preds['Predicted'] = y_pred

    # Save the DataFrame to a CSV file
    output_file = f'results/{model_type}/{model_name}/test_data_with_predictions.csv'
    test_data_with_preds.to_csv(output_file, index=False)


def save_confusion_matrix(model_type: str, model_name: str, cm: np.array,
                          y_test: pd.Series | np.ndarray):
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
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    plt.tight_layout()
    plt.savefig(f'results/{model_type}/{model_name}/confusion_matrix.png')
    plt.close()

