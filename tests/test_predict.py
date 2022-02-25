from svm_model import predict, config

from sklearn.metrics import accuracy_score
# for benchmark test, import the other model's predict function

# There are skl scoring metrics that do not automatically encode textual labels
# e.g. rocaucscore, but skl accuracy works
def test_prediction_accuracy(raw_training_data, sample_input_data):    
    # Given
    input_df = raw_training_data.drop(config.TARGET, axis=1)
    output_df = raw_training_data[config.TARGET]
    predictions = predict.make_prediction(input_data=input_df)
    
    # When
    accuracy = accuracy_score(y_true = output_df.values, y_pred = predictions)

    # Then
    assert accuracy > 0.90
    
'''
def test_prediction_quality_against_another_model(raw_training_data, sample_input_data):
    
    input_df = raw_training_data.drop(config.TARGET, axis=1)
    output_df = raw_training_data[config.TARGET]
    current_predictions = predict.make_prediction(input_data=input_df)
    alternative_predictions = alt_make_prediction(input_data=input_df)

    # When
    current_mse = mean_squared_error(y_true=output_df.values, y_pred=current_predictions["predictions"])

    alternative_mse = mean_squared_error(y_true=output_df.values, y_pred=alternative_predictions["predictions"])

    # Then
    assert current_mse < alternative_mse
'''    
    
'''
def test_reproducibility():
    
    Potential test for reproducibility compared with research phase Jupyter.
    pass

'''
