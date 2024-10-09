import pandas as pd




if __name__ == "__main__":
  df = pd.read_table('train.txt', names=['title', 'from', 'genre', 'director', 'plot'])
  print(df)