from svm_model import pipeline
from svm_model import config
from svm_model.processing.validation import validate_inputs


'''
def test_pipeline_transforms_email_data(pipeline_inputs):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    # We use the scikit-learn Pipeline private method `_fit` which is called
    # by the `fit` method, since this allows us to access the transformed
    # dataframe. For other models we could use the `transform` method, but
    # the GradientBoostingRegressor does not have a `transform` method.
    X_transformed, _ = pipeline.svm_pipe._fit(X_train, y_train)

    # FIX THIS
    assert (True)
'''

def test_pipeline_predict_takes_validated_input(pipeline_inputs, sample_input_data):
    # Given
    X_train, X_test, y_train, y_test = pipeline_inputs
    pipeline.svm_pipe.fit(X_train, y_train)

    # When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)
    predictions = pipeline.svm_pipe.predict(validated_inputs[config.FEATURES])

    # Then
    assert predictions is not None
    assert errors is None
