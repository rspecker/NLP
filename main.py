import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.preproc import preproc
from preprocessing.tfidf import create_train_df_tfidf

if __name__ == "__main__":
    # Import data
    df = pd.read_table('train.txt',
                       names=['title', 'from', 'genre', 'director', 'plot'])
    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=0)

    print("Before preproc: ", df['plot'][0])
    print("After preproc: ", preproc(df['plot'][0]))

    create_train_df_tfidf(df)
