import config
import pandas as pd

from data_management import load_pipeline, load_dataset


pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}.pkl"
_svm_pipe = load_pipeline(file_name = pipeline_file_name)


def make_prediction(*, input_data) -> dict:
    """Make a prediction using the saved model pipeline."""

    data = pd.DataFrame(input_data)

    prediction = _svm_pipe.predict(data[config.FEATURES])

    print("Test accuracy is: " + 
            str(_svm_pipe.score(prediction, data[config.TARGET]) * 100) + "%")

if __name__ == "__main__":
    make_prediction(input_data = load_dataset(file_name = config.TESTING_DATA_FILE))