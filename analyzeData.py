import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


def plot_genre_frequency(data):
  genre_cnt = data['genre'].value_counts()
  genre_cnt.plot(kind='bar', color='skyblue')
  plt.title('Frequency of genres')
  plt.xlabel('Genre')
  plt.ylabel('Frequency')
  plt.tight_layout()
  plt.savefig('plots/freq_genres.pdf')
  # plt.show()
  plt.clf()

def plot_plot_length_distr(data):
  data['length_plot'] = data['plot'].str.len()
  data['length_plot_words'] = data['plot'].str.count(' ') + 1

  print(data['length_plot_words'].mean())
  exit()

  data['length_plot'].plot.kde()
  plt.xlabel('length')
  plt.ylabel('Density')
  plt.title('Distribution of plot length')
  plt.savefig('plots/distr_plot_length.pdf')
  # plt.show()
  plt.clf()


def compare_distr_test_output(data):
  label_encoder = LabelEncoder()

  data['actual_encoded'] = label_encoder.fit_transform(data['actual'])
  data['predicted_encoded'] = label_encoder.fit_transform(data['predicted'])



  sns.histplot(data['actual_encoded'], color='blue', label='Actual', kde=True, stat="density", alpha=0.5)
  sns.histplot(data['predicted_encoded'], color='red', label='Predicted', kde=True, stat="density", alpha=0.5)

  plt.title('Distribution of Actual vs Predicted Values', fontsize=16)
  plt.xlabel('Value', fontsize=14)
  plt.ylabel('Density', fontsize=14)
  plt.legend(title='Category', fontsize=12, title_fontsize=12)
  plt.tight_layout()
  plt.savefig('plots/distr_act_pred_bar.pdf')
  # plt.show()

  plt.clf()

  order = sorted(data['actual'].unique())

  fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

  sns.countplot(data=data, x='actual', ax=axes[0], palette='Blues', order=order)
  axes[0].set_title('Actual Genre Frequency', fontsize=16)
  axes[0].set_xlabel('Actual Labels', fontsize=14)
  axes[0].set_ylabel('Count', fontsize=14)

  sns.countplot(data=data, x='predicted', ax=axes[1], palette='Reds', order=order)
  axes[1].set_title('Predicted Genre Frequency', fontsize=16)
  axes[1].set_xlabel('Predicted Labels', fontsize=14)
  axes[1].set_ylabel('') 


  plt.tight_layout()
  plt.savefig('plots/freq_act_pred.pdf')
  # plt.show()
  plt.clf()

  data[['actual_encoded', 'predicted_encoded']].plot.kde()
  plt.title('Distributions of predicted/actual')
  plt.savefig('plots/distr_act_pred.pdf')
  # plt.show()
  plt.clf()

if __name__ == '__main__':
  os.makedirs('plots', exist_ok=True)
  df_train = pd.read_table('train.txt',
                       names=['title', 'from', 'genre', 'director', 'plot'])
  df_train.to_csv('train.csv')
  plot_genre_frequency(df_train)
  plot_plot_length_distr(df_train)
  df_res = pd.read_csv('results/information_retrieval.csv')
  compare_distr_test_output(df_res)

