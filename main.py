import pandas as pd
from preprocessing.preproc import preproc




if __name__ == "__main__":
  df = pd.read_table('train.txt', names=['title', 'from', 'genre', 'director', 'plot'])
  #print(df)

  print("Before preproc: ", df['plot'][0])
  print("After preproc: ", preproc(df['plot'][0]))
