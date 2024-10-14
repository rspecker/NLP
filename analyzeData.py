import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def plotGenreFrequency(data):
  genre_cnt = data['genre'].value_counts()
  genre_cnt.plot(kind='bar', color='skyblue')
  plt.title('Freequency of genres')
  plt.xlabel('Genre')
  plt.ylabel('Frequency')
  plt.show()
  plt.clf()


def compareDistrTestOutput(data):
  label_encoder = LabelEncoder()

  data['actual_encoded'] = label_encoder.fit_transform(data['actual'])
  ddata['predicted_encoded'] = label_encoder.fit_transform(data['predicted'])



  sns.histplot(data['actual_encoded'], color='blue', label='Actual', kde=True, stat="density", alpha=0.5)
  sns.histplot(data['predicted_encoded'], color='red', label='Predicted', kde=True, stat="density", alpha=0.5)

  plt.title('Numerical Distribution of Actual vs Predicted Values', fontsize=16)
  plt.xlabel('Value', fontsize=14)
  plt.ylabel('Density', fontsize=14)
  plt.legend(title='Category', fontsize=12, title_fontsize=12)
  plt.tight_layout()
  plt.show()

  plt.clf()

  fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

  sns.countplot(data=data, x='actual', ax=axes[0], palette='Blues')
  axes[0].set_title('Actual Distribution', fontsize=16)
  axes[0].set_xlabel('Actual Labels', fontsize=14)
  axes[0].set_ylabel('Count', fontsize=14)

  sns.countplot(data=data, x='predicted', ax=axes[1], palette='Reds')
  axes[1].set_title('Predicted Distribution', fontsize=16)
  axes[1].set_xlabel('Predicted Labels', fontsize=14)
  axes[1].set_ylabel('')  # Hide y-label for the second plot

# # Adjust layout
  plt.tight_layout()
  plt.show()
  plt.clf()


if __name__ == '__main__':
  df = pd.read_table('train.txt',
                       names=['title', 'from', 'genre', 'director', 'plot'])
  df.to_csv('train.csv')
  plotGenreFrequency(df)
  df = pd.read_csv('res_IR.csv')
  compareDistrTestOutput(df)
