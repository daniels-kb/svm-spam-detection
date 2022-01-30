import pytest
from sklearn.model_selection import train_test_split

from svm_model import config
from svm_model.processing.data_management import load_dataset


@pytest.fixture(scope="session")
def pipeline_inputs():
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # Divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],  # predictors
        data[config.TARGET],
        test_size=config.TRAIN_TEST_SPLIT,
        # setting the random seed here for reproducibility
        random_state=config.SEED_VALUE,
    )

    return X_train, X_test, y_train, y_test


@pytest.fixture()
def raw_training_data():
    # For larger datasets, here we would use a testing sub-sample.
    return load_dataset(file_name=config.TRAINING_DATA_FILE)


@pytest.fixture()
def sample_input_data():
    return load_dataset(file_name=config.TESTING_DATA_FILE)
