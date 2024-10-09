import pandas as pd
from preprocessing.preproc import preproc
import time


def create_train_df(data):
  X = []
  for row in data.values.tolist():
    X_row = []
    for column in range(len(data.columns)):
      X_row.extend(preproc(row[column]))
    X.append(X_row)
  return X
      


if __name__ == "__main__":
  df = pd.read_table('train.txt', names=['title', 'from', 'genre', 'director', 'plot'])
  #print(df)

  print("Before preproc: ", df['plot'][0])
  print("After preproc: ", preproc(df['plot'][0]))

  start = time.time()
  print("hello")
  create_train_df(df)
  end = time.time()
  print(end - start)
  


