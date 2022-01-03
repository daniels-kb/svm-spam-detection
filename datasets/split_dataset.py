import pandas as pd
import config

from data_management import load_dataset
from sklearn.model_selection import train_test_split

data = load_dataset(file_name=config.DATASET_FILE)

# divide train and test
split = int(config.TRAIN_TEST_SPLIT * len(data))
X = data[split:]
y = data[:split]

X.to_csv('train.csv', index=False)
y.to_csv('test.csv', index=False)