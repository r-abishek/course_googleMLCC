# TensorFlow Notes

## From MLCC Google

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








### A few tf API functions
* tf.add()
* tf.constant()
* tf.global_variables_initializer()
* tf.Graph()
	* tf.Graph().as_default()
* tf.matmul()
* tf.reshape()
* tf.Session()
	* tf.Session.run()
* tf.Variable()
	* tf.Variable.assign()
	* tf.Variable.eval() (inside a session)
* tf.zeros()

