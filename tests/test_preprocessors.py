from svm_model import config
from svm_model.processing import preprocessors as pp

#FAILING 
def test_all_preprocessors(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    transformer = [pp.HyperlinkEncoder(), pp.SingleCharacterEncoder(), pp.StopwordRemover(), pp.WordStemmer()]

    # When
    for t in transformer:
        X_transformed = t.transform(X_train)

    # Check words inside emails from X_transformed were indeed transformed. Use iloc?
    # CONSIDER SPLITTING ASSERTION below into multiple
    assert X_transformed.isalpha()
    assert any(item in ["?", "!", "@", ".", "$", "ed ", "ly ", "ity ", set(stopwords.words('english'))] for item in X_transformed) 
