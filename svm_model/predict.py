import pandas as pd

from svm_model import config
from svm_model.processing.data_management import load_pipeline, load_dataset
from svm_model.processing.validation import validate_inputs


pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl"
_svm_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data = data)
    predictions = None
    
    if not errors:
        predictions = _svm_pipe.predict(X = validated_data[config.FEATURES])
        
    return predictions
    
# run predict.py to check accuracy
if __name__ == "__main__":
    print("Test accuracy is: " + 
            str(_svm_pipe.score(
            make_prediction(input_data = load_dataset(file_name = config.TESTING_DATA_FILE)), 
            data[config.TARGET]) * 100) + "%")