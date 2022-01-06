import preprocessors as pp
import config

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

svm_pipe = Pipeline(
    [
        ('hyperlink_encoder', pp.HyperlinkEncoder()),
        ('single_character_encoder', pp.SingleCharacterEncoder()),
        ('stopword_remover', pp.StopwordRemover()),
        ('word_stemmer', pp.WordStemmer()),
        
        ('vectoriser', TfidfVectorizer()),
        
        ('Support_Vector_Machine', svm.SVC(C = 10, gamma = 0.01, kernel = 'rbf'))
    ]
)
