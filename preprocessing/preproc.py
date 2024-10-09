import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt_tab')


def tokenize(text):
  tokenizer = RegexpTokenizer(r'\w+')
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

