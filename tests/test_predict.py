from svm_model import predict, config

from sklearn.metrics import mean_squared_error
# for benchmark, import other model's predict function


#FAILING
def test_make_predictions(raw_training_data, sample_input_data):
    pass
    
    input_df = raw_training_data.drop(config.TARGET, axis=1)
    output_df = raw_training_data[config.TARGET]
    current_predictions = predict.make_prediction(input_data=input_df)
    
    # When
    current_mse = mean_squared_error(y_true=output_df.values, y_pred=current_predictions)

    # Then
    assert current_mse > 0.9 
    

def test_prediction_quality_against_another_model(raw_training_data, sample_input_data):
    pass
    
    #input_df = raw_training_data.drop(config.TARGET, axis=1)
    #output_df = raw_training_data[config.TARGET]
    #current_predictions = predict.make_prediction(input_data=input_df)
    #alternative_predictions = alt_make_prediction(input_data=input_df)

    # When
    #current_mse = mean_squared_error(y_true=output_df.values, y_pred=current_predictions["predictions"])

    #alternative_mse = mean_squared_error(y_true=output_df.values, y_pred=alternative_predictions["predictions"])

    # Then
    #assert current_mse < alternative_mse

def test_reproducibility():
    """
    Potential test for reproducibility compared with research phase Jupyter.
    """
    pass

