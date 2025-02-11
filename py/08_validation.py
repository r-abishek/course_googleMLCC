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

# Read Dataframe
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep = ",")

# Modularized Preprocessing of features
def preprocess_features(california_housing_dataframe):
    """
    Prepares input features from California housing data set.
    Args:
        california_housing_dataframe: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
        A DataFrame that contains the features to be used for the model, including
        synthetic features.
    """
    selected_features = california_housing_dataframe[
        ["latitude",
        "longitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"]
    ]
    processed_features = selected_features.copy()

    # Created a synthetic feature
    processed_features["rooms_per_person"] = (
        california_housing_dataframe["total_rooms"] / 
        california_housing_dataframe["population"]
    )
    return processed_features

# Modularized Preprocessing of targets
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

    # Scale the target to be in units of thousands of dollars
    output_targets["median_house_value"] = (
        california_housing_dataframe["median_house_value"] / 1000.0
    )

    return output_targets

# Choose the first 12000 examples out of the 17000 for the training set
training_examples = preprocess_features(california_housing_dataframe.head(12000))
print(training_examples.describe())
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
print(training_targets.describe())


# Choose the last 5000 examples out of the 17000 for the validation set
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
print(validation_examples.describe())
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
print(validation_targets.describe())

# Plot latitude and longitude vs median house value
# Plot 1 with separated validation and training data

print("California Median House Prices Color Chart:")

plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_examples["longitude"],
            validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()

plt.show()

# Plot 2 with combined validation and training data

print("California Median House Prices Color Chart:")

plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data + Training Data")
ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_examples["longitude"],
            validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()

plt.show()

# Input Function
def my_input_fn(features, targets, batch_size = 1, shuffle = True, num_epochs = None):
    """
    Trains a linear regression model of multiple features.
      Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays
    features = {key : np.array(value) for key, value in dict(features).items()}

    # Construct a dataset and configure batching / repeating
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if required
    if shuffle:
        ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

# Function to configure feature columns
def construct_feature_columns(input_features):
  """
  Construct the TensorFlow Feature Columns.
  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """ 
  return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """
    Trains a linear regression model of multiple features.
    
    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.
    
    Args:
        learning_rate: A `float`, the learning rate.
        steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataframe` to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataframe` to use as target for validation.
        
    Returns:
        A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns = construct_feature_columns(training_examples),
        optimizer = my_optimizer
    )

    # Create input functions
    training_input_fn = lambda : my_input_fn(
        training_examples,
        training_targets["median_house_value"],
        batch_size = batch_size
    )
    predict_training_input_fn = lambda : my_input_fn(
        training_examples,
        training_targets["median_house_value"],
        shuffle = False,
        num_epochs = 1
    )
    predict_validation_input_fn = lambda : my_input_fn(
        validation_examples,
        validation_targets["median_house_value"],
        shuffle = False,
        num_epochs = 1
    )

    # Train the model, but do so inside a loop to periodically assess loss metrics
    print("Training Model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state
        linear_regressor.train(
            input_fn = training_input_fn,
            steps = steps_per_period
        )

        # Compute training predicitons
        training_predictions = linear_regressor.predict(input_fn = predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        # Compute validation predicitons
        validation_predictions = linear_regressor.predict(input_fn = predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute Training and Validation Loss
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets)
        )
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets)
        )

        # Occasionally print the current loss
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))

        # Add the loss metrics from this period to our list
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    
    print("Model Training Finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()

    plt.show()

    return linear_regressor

# Train model and use training and validation data
linear_regressor = train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

# Load Test Set
california_housing_test_data = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")

# Choose the test set
test_examples = preprocess_features(california_housing_test_data)
print(test_examples.describe())
test_targets = preprocess_targets(california_housing_test_data)
print(test_targets.describe())

predict_test_input_fn = lambda : my_input_fn(
    test_examples,
    test_targets["median_house_value"],
    num_epochs = 1,
    shuffle = False
)

test_predictions = linear_regressor.predict(input_fn = predict_test_input_fn)
test_predictions = np.array([item['predictions'][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets)
)

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
