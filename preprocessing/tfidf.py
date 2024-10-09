from preprocessing.preproc import preproc
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def create_preprocesssed_dataset(data: pd.DataFrame):
  X = []
  for row in data.values.tolist():
    X_row = []
    for column in range(len(data.columns)):
      X_row.extend(preproc(row[column]))
    X.append(X_row)
  return X


def own_tokenizer(data): # identiy tokenizer
  return data

def create_train_data_tfidf(data: pd.DataFrame) -> tuple[TfidfVectorizer, ...]:
  vectorizer = TfidfVectorizer(tokenizer=own_tokenizer, lowercase=False, stop_words=None)
  X = vectorizer.fit_transform(create_preprocesssed_dataset(data))
  return (vectorizer, X)
      