import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download("stopwords")

# Convert email addresses and URLs to proper words
class HyperlinkEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.url_re = re.compile(r'(http|https)://[^\s]*')
        self.mail_re = re.compile(r'[^\s]+@[^\s]+[.][^\s]+')
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # convert email addresses to 'emailaddr' and URLs to 'httpaddr'
        for email in X:
            email = self.url_re.sub(r' httpaddr ', email) 
            email = self.mail_re.sub(r' emailaddr ', email)
       
        return X
        
# Deal with whitespace and other encode non alphabetic characters
class SingleCharacterEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.number_re = re.compile(r'[0-9]+')
        self.dollar_re = re.compile(r'[$]')
        self.exclammark_re = re.compile(r'[!]')
        self.questmark_re = re.compile(r'[?]')
        self.punct_re = re.compile(r'([^\w\s]+)|([_-]+)')
    
        return self

    def transform(self, X):
        X = X.copy()
        
        for email in X:
            # convert numbers to 'number'
            email = self.number_re.sub(r' number ', email)

            # convert $, ! and ? to proper words
            email = self.dollar_re.sub(r' dollar ', email)
            email = self.exclammark_re.sub(r' exclammark ', email)
            email = self.questmark_re.sub(r' questmark ', email)

            # convert other punctuation to whitespace
            email = self.punct_re.sub(r' ', email)
            
            # remove trailing and leading whitespace
            email = email.strip(' ')

        return X
        
# Remove English language stopwords
class StopwordRemover(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        self.stopwords_re = re.compile(r'\b(' + r'|'.join(nltk.corpus.stopwords.words('english')) + r')\b\s*')
        
        return self

    def transform(self, X):
        X = X.copy()

        for email in X:
            email = self.stopwords_re.sub('', email)
        
        return X
    
# Stem words of endings e.g. 'depositing/deposited/deposit' all get converted to 'deposit'
class WordStemmer(BaseEstimator, TransformerMixin):

    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        for email in X:
            emailWords = email.split(' ')
            stemmer = nltk.stem.snowball.SnowballStemmer('english')
            stemWords = [stemmer.stem(word) for word in emailWords]
            email = ' '.join(stemWords)
        
        return X