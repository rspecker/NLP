import pandas as pd
from preprocessing.preproc import create_preprocesssed_dataset
from preprocessing.tfidf import create_train_data_tfidf
from utils import create_train_test_sets

if __name__ == "__main__":
    # Import data
    df = pd.read_table('train.txt',
                       names=['title', 'from', 'genre', 'director', 'plot'])
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = create_train_test_sets(
        df, test_size=0.2, random_state=0, y_column='genre'
    )

    x_train = create_train_data_tfidf(x_train)
    x_test = create_train_data_tfidf(x_test)

    y_train = create_preprocesssed_dataset(y_train)
    y_test = create_preprocesssed_dataset(y_test)
