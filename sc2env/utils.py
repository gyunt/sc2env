import tensorflow as tf


def save_graph(path, sess=None):
    sess = sess or tf.get_default_session()
    writer = tf.summary.FileWriter(path, sess.graph)
    writer.flush()
