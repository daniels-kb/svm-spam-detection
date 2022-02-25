import nltk
import pandas as pd
from svm_model import config
from svm_model.processing import preprocessors as pp

# Check transformers work as intended. If they do, no one item in test_strings will be found in any emails.
def test_all_preprocessors(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    transformer = [pp.HyperlinkEncoder(), pp.SingleCharacterEncoder(), pp.StopwordRemover(), pp.WordStemmer()]
    
    test_strings = nltk.corpus.stopwords.words('english')
    test_strings.extend(["?", "!", "@", ".", "$", "ed ", "ly ", "ity"])

    # When
    for t in transformer:
        X_transformed = t.fit_transform(X_train)

    assert isinstance(X_transformed, pd.Series)
    assert any(X_transformed.str.isalpha())
    assert (item in test_strings for item in X_transformed)