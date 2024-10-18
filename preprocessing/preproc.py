import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import spacy
from spacy.cli import download
import re

try:
  nlp = spacy.load("en_core_web_sm")
except:
  download('en_core_web_sm')
  nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords')
nltk.download('punkt_tab')


def tokenize(text):
  text = re.sub(r'\.([a-zA-Z])', r'\1', text) # Ph.D. -> PhD. | S.H.I.E.L.D -> SHIELD
  tokenizer = RegexpTokenizer(r'[\w\-]+') #hyphen due to sci-fi
  return tokenizer.tokenize(text)

def to_lower(tokens):
  return [t.lower() for t in tokens]

def stopwords_rem(tokens, lowercased=True, language='english'): 
  sw = set(stopwords.words(language))
  if lowercased:
    return [t for t in tokens if t not in sw]
  else:
    return [t for t in tokens if t.lower() not in sw]
  
def stemming(tokens):
  ps = PorterStemmer()
  return [ps.stem(word = t, to_lowercase=False) for t in tokens]

def preproc(text, lower=True, stopwords=True, lang='english', stemm=True):
  tokens = tokenize(text)
  if lower:
    tokens = to_lower(tokens)
  if stopwords:
    tokens = stopwords_rem(tokens, lowercased=lower, language=lang)
  if stemm:
    tokens = (stemming(tokens))

  return tokens

def create_preprocesssed_dataset(data: pd.DataFrame):
  column_nr = 0
  try:
    column_nr = len(data.columns)
  except:
    X = []    
    for el_row in data.values.tolist():
      X.append(preproc(el_row))
    return X


  X = []
  for row in data.values.tolist():
    X_row = []
    for column in range(column_nr):
      X_row.extend(preproc(row[column]))
    X.append(X_row)
  return X

def extract_entities(text, special_labels = None):
  doc = nlp(text)
  entities = None
  if special_labels != None:
    entities = [ent.text for ent in doc.ents if ent.label_ in special_labels]
  else:
    entities = [ent.text for ent in doc.ents]
  return entities

def create_named_entity_column(data):
  data['persons'] = data['plot'].apply(extract_entities, special_labels=['PERSON'])
  return data


