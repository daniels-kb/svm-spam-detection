import pipeline
import config

from sklearn.model_selection import train_test_split
from data_management import load_dataset, save_pipeline


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.DATASET_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size = config.TRAIN_TEST_SPLIT,
        shuffle = True,
        random_state = config.SEED_VALUE)
    
    pipeline.svm_pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist = pipeline.svm_pipe)


if __name__ == '__main__':
    run_training()
