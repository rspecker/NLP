import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_sets(df: pd.DataFrame, test_size=0.2, random_state=0,
                         y_column: str='genre'):
    """
    Splits a DataFrame into training and testing sets, while also separating features (X) and target (y).

    Parameters:
    df: The input DataFrame to split.
    test_size: The proportion of the dataset to include in the test split
        (default is 0.2).
    random_state (int): The seed used by the random number generator
        (default is 0).
    y_column: The name of the target column (default is 'genre').

    Returns:
        X_train, X_test (features), y_train, y_test (target).
    """
    # Separate features (X) and target (y)
    x = df.drop(columns=[y_column])
    y = df[y_column]

    # Split the data into training and testing sets. When splitting the data,
    # stratify the samples to ensure that the proportion of each class is
    # preserved in both sets.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return x_train, x_test, y_train, y_test
