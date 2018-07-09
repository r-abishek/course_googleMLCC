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
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep = ", ")

# Randomize the data
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)

# Scale median_house_value to be in thousands
california_housing_dataframe["median_house_value"] /= 1000.0
california_housing_dataframe

# Print a summary
california_housing_dataframe.describe()

# Define the input feature: total_rooms
my_feature = california_housing_dataframe[["total_rooms"]]

# Configure a numeric feature column for total_rooms
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

# Define the label or target as median_house_value
targets = california_housing_dataframe["median_house_value"]

# Use gradient descent as the optimizer for training the model
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)

# Clip gradients in the optimizer to avoid exploding gradients
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns = feature_columns,
    optimizer = my_optimizer
)

# Function to train a linear regression model of one feature
def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
    """
    Args:
        features: pandas dataframe of features
        targets: pandas dataframe of targets
        batch_size = size of batches
        shuffle = True/False
        num_epochs = Number of epochs
    Returns:
        Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays
    features = {key:np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching / repeating
    ds = Dataset.from_tensor_slices((features, targets)) # 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# Training the model
_ = linear_regressor.train(
    input_fn = lambda : my_input_fn(my_feature, targets),
    steps = 100
)

# Creating an input function for predictions
prediction_input_fn = lambda : my_input_fn(my_feature, targets, num_epochs = 1, shuffle = False)

# Call predict() on the linear_regressor to make predictions
predictions = linear_regressor.predict(input_fn = prediction_input_fn)

# Format predicitons as a Numpy array, so that we can calcualte error metrics
predictions = np.array([item['predictions'][0] for item in predictions])

# Print MSE and RMSE
mse = metrics.mean_squared_error(predictions, targets)
rmse = math.sqrt(mse)
print "Mean Squared Error (on training data): %0.3f" % mse
print "Root Mean Squared Error (on training data): %0.3f" % rmse
