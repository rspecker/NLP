from sklearn.metrics.pairwise import cosine_similarity
from preprocessing.preproc import create_preprocesssed_dataset
from preprocessing.tfidf import create_train_data_tfidf, query_vectorizer, remove_words_test
import numpy as np
import pandas as pd

def predict(X_knowledge, Y_knowledge, x_sample):
  cosineSimilarities = cosine_similarity(x_sample, X_knowledge).flatten()

  index = np.argmax(cosineSimilarities)
  return Y_knowledge[index]

def score(X_knowledge, X_test, Y_knowledge, Y_test):
  correct_ctr = 0

  X_knowledge_only_pre = create_preprocesssed_dataset(X_knowledge)
  (vectorizer, X_knowledge) = create_train_data_tfidf(X_knowledge, max_features=None) 
  
  X_test = create_preprocesssed_dataset(X_test)
  Y_test = Y_test.values.tolist()

  Y_knowledge = Y_knowledge.values.tolist()

  X_test = remove_words_test(vectorizer.vocabulary_.keys(), X_test)

  df = pd.DataFrame(columns=['predicted', 'actual'])
  for (x_sample, y_sample) in zip(X_test, Y_test):
    x_sample = query_vectorizer(X_knowledge_only_pre, [x_sample])
    res = predict(X_knowledge, Y_knowledge, x_sample)
    
    if res == y_sample:
      correct_ctr += 1

    df.loc[len(df)] = [res, y_sample]
  print("accuracy score: ", correct_ctr/len(X_test))
  df.to_csv('information_retrieval.csv')


  
