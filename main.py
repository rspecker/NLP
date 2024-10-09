import pandas as pd
from preprocessing.tfidf import create_train_df_tfidf


if __name__ == "__main__":
  df = pd.read_table('train.txt', names=['title', 'from', 'genre', 'director', 'plot'])
  #print(df)

  print("Before preproc: ", df['plot'][0])
  print("After preproc: ", preproc(df['plot'][0]))

 
  create_train_df_tfidf(df)
 
  


