from preprocessing.preproc import create_preprocesssed_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def own_tokenizer(data): # identiy tokenizer
  return data

def remove_words_test(voc, X):
  X_new = []
  for row in X:
    X_new.append([w for w in row if w in voc])
  return X_new

def query_vectorizer(data, query):
  vectorizer = TfidfVectorizer(tokenizer=own_tokenizer, lowercase=False, stop_words=None)  
  queryV = vectorizer.fit(data)
  return queryV.transform(query)

def create_train_data_tfidf(data: pd.DataFrame, vectorizer=None, max_features=10000) -> tuple[TfidfVectorizer, ...]:
  if vectorizer == None:
    vectorizer = TfidfVectorizer(tokenizer=own_tokenizer, lowercase=False, stop_words=None, max_features=max_features)
    vectorizer = vectorizer.fit(create_preprocesssed_dataset(data))
    X = vectorizer.transform(create_preprocesssed_dataset(data))
  else:
    X = create_preprocesssed_dataset(data)
    X = remove_words_test(vectorizer.vocabulary_.keys(), X)
    X = vectorizer.transform(X)
    
  return (vectorizer, X)
      