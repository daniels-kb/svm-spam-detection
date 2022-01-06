import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

DATASET_FILE = "emails.csv"

TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"

FEATURES = "EmailText"
TARGET = "Label"

PIPELINE_NAME = "vector_machine.pkl"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

TRAIN_TEST_SPLIT = 0.25

SEED_VALUE = 0
