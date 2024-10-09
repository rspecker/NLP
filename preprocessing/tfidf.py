from preprocessing.preproc import preproc

def create_train_df_tfidf(data):
  X = []
  for row in data.values.tolist():
    X_row = []
    for column in range(len(data.columns)):
      X_row.extend(preproc(row[column]))
    X.append(X_row)
  return X
      