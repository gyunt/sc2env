import tensorflow as tf
from baselines.common.tf_util import get_session


def save_graph(path):
    writer = tf.summary.FileWriter(path, get_session().graph)
    writer.flush()
