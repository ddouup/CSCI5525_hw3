"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y that are Gaussian distributed, zero mean unit variance.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond() and modify example code below
###############################################################################

x = tf.random_normal(shape=[], mean=0.0, stddev=1.0)
y = tf.random_normal(shape=[], mean=0.0, stddev=1.0)
out = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))

with tf.Session() as sess:
    print('1a:')
    print(out.eval())
    print()

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

x = tf.random_uniform(shape=[], minval=0, maxval=None)
y = tf.random_uniform(shape=[], minval=0, maxval=None)
out = tf.case({tf.less(x, y): lambda: tf.add(x, y), tf.greater(x,y): lambda: tf.subtract(x, y)}, default= lambda: tf.constant(0, dtype=tf.float32), exclusive = True)

with tf.Session() as sess:
    print('1b:')
    print(out.eval())
    print()

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.constant(0, shape=[2, 3])
out = tf.equal(x, y)

with tf.Session() as sess:
    print('1c:')
    print(out.eval())
    print()

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
                30.97266006,  26.67541885,  38.08450317,  20.74983215,
                34.94445419,  34.45999146,  29.06485367,  36.01657104,
                27.88236427,  20.56035233,  30.20379066,  29.51215172,
                33.71149445,  28.59134293,  36.05556488,  28.66994858])
indices = tf.where(x>30)
out = tf.gather(x, indices)

with tf.Session() as sess:
    print('1d:')
    print(indices.eval())
    print(out.eval())
    print()

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

out = tf.diag(tf.range(1, 7, 1))

with tf.Session() as sess:
    print('1e')
    print(out.eval())
    print()

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

x = tf.random_normal(shape=[10,10], mean=0, stddev=1.0)
out = tf.matrix_determinant(x)

with tf.Session() as sess:
    print('1f:')
    print(out.eval())
    print()

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
out, idx= tf.unique(x)

with tf.Session() as sess:
    print('1g:')
    print(out.eval())
    print(idx.eval())
    print()

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution. Compute
# - The mean squared error of (x - y) 
###############################################################################

x = tf.random_normal(shape=[300], mean=0, stddev=1.0)
y = tf.random_normal(shape=[300], mean=0, stddev=1.0)
out = tf.losses.mean_squared_error(x, y)

with tf.Session() as sess:
    print('1h:')
    print(out.eval())
    print()