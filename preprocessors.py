import re
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

# Convert email addresses and URLs to proper words
class HyperlinkEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.url_regex = re.compile(r'(http|https)://[^\s]*')
        self.mail_regex = re.compile(r'[^\s]+@[^\s]+[.][^\s]+')
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # convert email addresses to 'emailaddr' and URLs to 'httpaddr'
        for email in X:
            email = self.url_regex.sub(r' httpaddr ', email) 
            email = self.mail_regex.sub(r' emailaddr ', email)
       
        return X
        
# Deal with whitespace and other encode non alphabetic characters
class SingleCharacterEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # convert numbers to 'number'
        X = re.sub(r'[0-9]+', r' number ', X)

        # convert $, ! and ? to proper words
        X = re.sub(r'[$]', r' dollar ', X)
        X = re.sub(r'[!]', r' exclammark ', X)
        X = re.sub(r'[?]', r' questmark ', X)

        # convert other punctuation to whitespace
        X = re.sub(r'([^\w\s]+)|([_-]+)', r' ', X)
        
        # remove trailing and leading whitespace
        X = X.strip(' ')
        return X
        
# Remove English language stopwords
class StopwordRemover(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.pattern = re.compile(r'\b(' + r'|'.join(nltk.corpus.stopwords.words('english')) + r')\b\s*')
        
        return self

    def transform(self, X):
        X = X.copy()
        X = self.pattern.sub('', X)
        
        return X
    
# Stem words of endings e.g. 'depositing/deposited/deposit' all get converted to 'deposit'   
class WordStemmer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.stemWords = []
        
        # perform word stemming
        emailWords = X.split(' ')
        stemmer = nltk.stem.snowball.SnowballStemmer('english')
        for word in emailWords:
            stemWords.append(stemmer.stem(word))
        
        return self

    def transform(self, X):
        X = X.copy()
        X = ' '.join(self.stemWords)
        
        return X