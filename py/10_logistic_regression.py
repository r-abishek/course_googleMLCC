from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Load Dataset
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

# Permutation
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)

def preprocess_features(california_housing_dataframe):
    """Prepares input features from California housing data set.
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """

    selected_features = california_housing_dataframe[
        [
            "latitude",
            "longitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income"
        ]
    ]
    processed_features = selected_features.copy()

    # Create synthetic feature
    processed_features = ["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] / 
        california_housing_dataframe["population"]
    )

    return processed_features

def preprocess_targets(california_housing_dataframe):
    """
    Prepares target features (i.e., labels) from California housing data set.
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()

    # Create a boolean categorical feature representing whether the median_house_value is above a set threshold
    output_targets["median_house_value_is_high"] = (
        california_housing_dataframe["median_house_value"] > 265000
    ).astype(float)

    return output_targets

# Choose the first 12000 (out of 17000) examples for training
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

# Choose the last 5000 (out of 17000) examples for validation
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# Double Checking
print("Training Examples Summary:")
print(training_examples.describe())
print("Training Targets Summary:")
print(training_targets.describe())
print("Validation Examples Summary:")
print(validation_examples.describe())
print("Validation Targets Summary:")
print(validation_targets.describe())
