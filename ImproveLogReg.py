import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1









with tf.name_scope('loss'):
    # you should know how to do this too
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='loss')

with tf.name_scope('optimizer'):
    #: define training op
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs/mnist_lstm', sess.graph)
    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch = sess.run([optimizer, loss],
                                 feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))