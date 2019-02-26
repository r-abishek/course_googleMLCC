# TensorFlow Notes from MLCC - Google

## Notes as code-snippets extracted from examples

### Basics
    import tensorflow as tf
    c = tf.constant('Hello, world!')
    with tf.Session() as sess:

### Variables and Constants
    x = tf.constant(5.2)
    y = tf.Variable([5])
    primes = tf.constant([2,3,5,7,11,13], dtype=tf.int32)

### Variables and Assigning
    y = tf.Variable([0])
    y = y.assign([5])

### Running a session after initializing variables (needs global variables initializer)
    with tf.Session() as sess:
    	initialization = tf.global_variables_initializer()
    	print y.eval()

### Creating a graph and using it as default before initiating a session
    g = tf.Graph()
    with g.as_default():
    	x = tf.constant(8, name="x_const")
    	y = tf.constant(5, name="y_const")
    	z = tf.constant(4, name="z_const")
    	my_sum = tf.add(x, y, name="x_y_sum")
    	my_sum = tf.add(my_sum, z, name="x_y_z_sum")
    	with tf.Session() as sess:
    		print my_sum.eval()

### Global variables initializer
    tf.global_variables_initializer().run()

### Using tf.group() - {not quite clear!}
    zs_ = zs * zs + xs
    not_diverged = tf.abs(zs_) < 4
    step = tf.group(
    	zs.assign(zs_),
    	ns.assign_add(tf.cast(not_diverged, tf.float32))
    )

### Using tf.reshape()
    with tf.Graph().as_default():
    	matrix = tf.constant([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)
    	reshaped_2x8_matrix = tf.reshape(matrix, [2,8])
    	reshaped_4x4_matrix = tf.reshape(matrix, [4,4])
    	reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])
    	one_dimensional_vector = tf.reshape(matrix, [16])
    	with tf.Session() as sess:
    		print "Original matrix (8x2):"
    		print matrix.eval()
		print "Reshaped matrix (2x8):"
    		print reshaped_2x8_matrix.eval()
    		print "Reshaped matrix (4x4):"
    		print reshaped_4x4_matrix.eval()
    		print "Reshaped 3-D tensor (2x2x4):"
    		print reshaped_2x2x4_tensor.eval()
    		print "1-D vector:"
    		print one_dimensional_vector.eval()`

### Defining a feature column and target for the model
    my_feature = california_housing_dataframe[["total_rooms"]]
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]
    targets = california_housing_dataframe["median_house_value"]

### Configuring LinearRegressor for the model
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

### Training a linear regressor model
    _ = linear_regressor.train(
    	input_fn = lambda:my_input_fn(my_feature, targets),
    	steps=100
    )

### Evaluate a linear regressor model
    prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
    predictions = linear_regressor.predict(input_fn=prediction_input_fn)
    predictions = np.array([item['predictions'][0] for item in predictions])
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
    print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)

### Retrieving weights and biases from training
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

### Using FtrlOptimizer for L1 regularization (for better results than standard gradient descent)
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
    	feature_columns=feature_columns,
    	optimizer=my_optimizer
    )

### Using tf.feature_column.bucketized_column()
    bucketized_latitude = tf.feature_column.bucketized_column(latitude, boundaries=get_quantile_based_boundaries(training_examples["latitude"], 10))
    bucketized_housing_median_age = tf.feature_column.bucketized_column(housing_median_age, boundaries=get_quantile_based_boundaries(training_examples["housing_median_age"], 7))

### Using tf.feature_column.crossed_column() to cross two or more features
    long_x_lat = tf.feature_column.crossed_column(set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

### Creating the DNNRegressor object
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
    	feature_columns=construct_feature_columns(training_examples),
    	hidden_units=hidden_units,
    	optimizer=my_optimizer
    )

### Training and predicting on the DNNRegressor model
    dnn_regressor.train(
    	input_fn=training_input_fn,
    	steps=steps_per_period
    )
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])    
    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

### Using tf.io, tf.data for parsing training and test data in an input pipeline 
		def _parse_function(record):
			features = {
				"terms": tf.VarLenFeature(dtype=tf.string), # terms are strings of varying lengths
				"labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32) # labels are 0 or 1
			}
			parsed_features = tf.parse_single_example(record, features)
			terms = parsed_features['terms'].values
			labels = parsed_features['labels']
			return  {'terms':terms}, labels

		# Create the Dataset object.
		ds = tf.data.TFRecordDataset(train_path)
		# Map features and labels with the parse function.
		ds = ds.map(_parse_function)
		# Pad and batch each field of the dataset to whatever size necessary
		ds = ds.padded_batch(25, ds.output_shapes)
		ds = ds.repeat(num_epochs)
		# Get next batch of data
		features, labels = ds.make_one_shot_iterator().get_next()

### Using tf.keras.layers, tf.keras.Model, tf.keras.optimizers, tf.keras.preprocessing.image

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

img_input = layers.Input(shape=(150, 150, 3))
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = Model(img_input, output)
model.summary()

model.compile(loss='binary_crossentropy',
	optimizer=RMSprop(lr=0.001),
	metrics=['acc']
)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,  # 1000 images = batch_size * steps
	verbose=2
)








## A few tf API functions
* tf
	* tf.constant()
	* tf.Graph
		* tf.Graph.as_default()
	* tf.reshape()
	* tf.Session()
		* tf.Session.as_default()
		* tf.Session.run()
	* tf.Variable()
		* tf.Variable.assign()
		* tf.Variable.eval() (inside a session)
	* tf.zeros
* tf.contrib
	* tf.contrib.estimator
		* tf.contrib.estimator.clip_gradients_by_norm()
* tf.data
	* tf.data.Dataset
		* tf.data.Dataset.batch()
		* tf.data.Dataset.from_tensor_slices()
		* tf.data.Dataset.make_one_shot_iterator()
		* tf.data.Dataset.repeat()
		* tf.data.Dataset.shuffle()
	* tf.data.Iterator
		* tf.data.Iterator.get_next()
	* tf.data.TFRecordDataset
		* tf.data.TFRecordDataset.make_one_shot_iterator()
		* tf.data.TFRecordDataset.map()
		* tf.data.TFRecordDataset.padded_batch()
		* tf.data.TFRecordDataset.repeat()
* tf.estimator
	* tf.estimator.LinearClassifier
		* tf.estimator.LinearClassifier.evaluate()
		* tf.estimator.LinearClassifier.predict()
		* tf.estimator.LinearClassifier.train()
	* tf.estimator.LinearRegressor
		* tf.estimator.LinearRegressor.predict()
		* tf.estimator.LinearRegressor.train()
	* tf.estimator.DNNClassifier
		* tf.estimator.DNNClassifier.get_variable_value()
		* tf.estimator.DNNClassifier.predict()
		* tf.estimator.DNNClassifier.train()
	* tf.estimator.DNNRegressor
		* tf.estimator.DNNRegressor.predict()
		* tf.estimator.DNNRegressor.train()
* tf.feature_column
	* tf.feature_column.bucketized_column()
	* tf.feature_column.categorical_column_with_vocabulary_list()
	* tf.feature_column.crossed_column()
	* tf.feature_column.embedding_column()
	* tf.feature_column.indicator_column()
	* tf.feature_column.numeric_column()
* tf.initializers
	* tf.initializers.global_variables() # or tf.global_variables_initializer()
* tf.io
	* tf.io.FixedLenFeature
	* tf.io.parse_single_example()
	* tf.io.VarLenFeature
* tf.keras
	* tf.keras.layers
		* tf.keras.layers.Conv2D()
		* tf.keras.layers.Dense()
		* tf.keras.layers.Flatten()
		* tf.keras.layers.MaxPooling2D()
	* tf.keras.models
		* tf.keras.models.Model
			* tf.keras.models.Model.compile()
			* tf.keras.models.Model.fit_generator()
			* tf.keras.models.Model.layers
			* tf.keras.models.Model.output
			* tf.keras.models.Model.summary()
	* tf.keras.optimizers
		* tf.keras.optimizers.Adagrad
		* tf.keras.optimizers.Adam
		* tf.keras.optimizers.RMSprop
	* tf.keras.preprocessing
		* tf.keras.preprocessing.image
			* tf.keras.preprocessing.image.ImageDataGenerator
				* tf.keras.preprocessing.image.ImageDataGenerator.flow()
				* tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory()
			* tf.keras.preprocessing.image.img_to_array
			* tf.keras.preprocessing.image.load_img

	* tf.keras.utils
		* tf.keras.utils.get_file()
* tf.linalg
	* tf.linalg.matmul() # tf.matmul()
* tf.logging
	* tf.logging.set_verbosity(tf.logging.ERROR)
* tf.manip
	* tf.manip.reshape()
* tf.math
	* tf.math.add() # Or tf.add()
* tf.train()
	* tf.train.AdagradOptimizer()
	* tf.train.FtrlOptimizer()
	* tf.train.GradientDescentOptimizer()


