import tensorflow as tf
def printf(x):
    with tf.Session() as sess:
     # We can also use 'c.eval()' here.
        print(sess.run(x))