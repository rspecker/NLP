import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.preproc import preproc
from preprocessing.tfidf import create_train_df_tfidf
from utils import create_train_test_sets

if __name__ == "__main__":
    # Import data
    df = pd.read_table('train.txt',
                       names=['title', 'from', 'genre', 'director', 'plot'])
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = create_train_test_sets(
        df, test_size=0.2, random_state=0, y_column='genre'
    )


    print("Before preproc: ", df['plot'][0])
    print("After preproc: ", preproc(df['plot'][0]))

    create_train_df_tfidf(df)
