import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops


def primary_caps(inputs, num_capsule, vec_len, kernel_size, stride=1, scope='PrimaryCapsuleLayer'):
    assert num_capsule % vec_len == 0

    with tf.variable_scope(scope):
        input_rank = inputs.get_shape().ndims

        if input_rank != 4:
            raise ValueError('PrimaryCaps not supported for input with rank', input_rank)

        net = tf.contrib.layers.conv2d(inputs,
                                       num_outputs=num_capsule,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding='valid',
                                       activation_fn=tf.nn.relu)
        net = tf.contrib.layers.flatten(net)
        net = tf.reshape(net, [-1, net.get_shape().as_list()[1] // vec_len, vec_len])
    return net


class CapsuleLayer(base.Layer):
    def __init__(self,
                 num_capsule,
                 vec_len,
                 weight_initializer=init_ops.random_normal_initializer(),
                 activity_regularizer=None,
                 trainable=True,
                 output_type='vector',  # 'vector' or 'scalar'
                 num_routing=1,
                 name=None,
                 **kwargs):
        super(CapsuleLayer, self).__init__(trainable=trainable, name=name,
                                           activity_regularizer=activity_regularizer,
                                           **kwargs)
        self.num_capsule = num_capsule
        self.vec_len = vec_len
        self.weight_initializer = weight_initializer
        self.input_spec = base.InputSpec(min_ndim=3, max_ndim=3)
        self.output_type = output_type
        self.num_routing = num_routing

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        _, previous_num_capsule, previous_vec_len = input_shape.as_list()

        self.previous_num_capsule = previous_num_capsule
        self.previous_vec_len = previous_vec_len
        self.B = self.add_variable('B',
                                   shape=[previous_num_capsule, self.num_capsule],
                                   initializer=tf.zeros_initializer(),
                                   dtype=self.dtype,
                                   trainable=True)

        self.T = self.add_variable('W',
                                   shape=(previous_num_capsule,
                                          self.num_capsule,
                                          previous_vec_len,
                                          self.vec_len),
                                   initializer=self.weight_initializer,
                                   dtype=self.dtype,
                                   trainable=True)

        self.biases = self.add_variable('biases',
                                        shape=(1, self.num_capsule, self.vec_len),
                                        initializer=tf.zeros_initializer(),
                                        dtype=self.dtype,
                                        trainable=True)

        self.built = True

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        T = self.T
        T = tf.expand_dims(T, axis=0)
        T = tf.tile(T, multiples=[batch_size, 1, 1, 1, 1])
        frobenium_norm = tf.norm(T, ord='fro', axis=[2, 3])

        assert T.get_shape().is_compatible_with(
            [None,
             self.previous_num_capsule,
             self.num_capsule,
             self.previous_vec_len,
             self.vec_len]), '%s, %s' % (T.get_shape(), [None,
                                                         self.previous_num_capsule,
                                                         self.num_capsule,
                                                         self.previous_vec_len,
                                                         self.vec_len])

        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        B = self.B
        U = inputs

        with tf.variable_scope('UHat'):
            U = tf.expand_dims(U, axis=2)
            U = tf.expand_dims(U, axis=2)
            U = tf.tile(U, multiples=[1, 1, self.num_capsule, 1, 1])
            U_hat = tf.squeeze(tf.matmul(U, T), axis=3)



            frobenium_norm = tf.expand_dims(frobenium_norm, axis=-1)
            frobenium_norm = tf.tile(frobenium_norm, multiples=[1, 1, 1, self.vec_len])

            print(U_hat.get_shape(), frobenium_norm.get_shape())
            exit()

            o = tf.multiply(frobenium_norm, U_hat)
        print(o.get_shape())
        assert (o.get_shape().is_compatible_with([None, self.previous_num_capsule, self.num_capsule, self.vec_len]))
        exit()

        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        U_hat_stopped = tf.stop_gradient(U_hat, name='stop_gradient')

        biases = tf.tile(self.biases, multiples=[batch_size, 1, 1])

        for i in range(self.num_routing):
            C = tf.nn.softmax(B)
            C = tf.expand_dims(C, axis=0)
            C = tf.expand_dims(C, axis=-1)
            C = tf.tile(C, multiples=[batch_size, 1, 1, self.vec_len])

            if i == self.num_routing - 1:
                S = tf.multiply(U_hat, C)
                S = tf.reduce_sum(S, axis=1) + biases
                assert (S.get_shape().is_compatible_with([None, self.num_capsule, self.vec_len]))

                V = self.squash(S)
                assert (V.get_shape().is_compatible_with(S.get_shape()))
            else:
                S = tf.multiply(U_hat_stopped, C)
                S = tf.reduce_sum(S, axis=1) + biases
                assert (S.get_shape().is_compatible_with([None, self.num_capsule, self.vec_len]))

                V = self.squash(S)
                assert (V.get_shape().is_compatible_with(S.get_shape()))

                multi_U_hat_V = self.update_b(U_hat_stopped, V)
                delta_B = tf.reduce_mean(multi_U_hat_V, axis=0)
                B = B + delta_B

        # self.B.assign(self.B + delta_B)
        # TODO: update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_b_op)

        if self.output_type == 'scalar':
            net = tf.sqrt(tf.reduce_sum(tf.square(V), axis=2))
        elif self.output_type == 'vector':
            net = V
        else:
            raise ValueError('output type should be \'scalar\' or \'vector\'.')

        return net

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        output_shape = [input_shape[0], self.previous_num_capsule, self.previous_vec_len]
        return output_shape

    @staticmethod
    def get_s(U_hat, C):
        with tf.variable_scope('GetS'):
            C = tf.expand_dims(C, axis=3)
            C = tf.tile(C, [1, 1, 1, tf.shape(U_hat)[-1]])

            S = tf.multiply(U_hat, C)
            S = tf.reduce_sum(S, axis=1)
            return S

    @staticmethod
    def squeeze(inputs):
        with tf.variable_scope('Squeeze'):
            norm = tf.norm(inputs, axis=0)
            print(norm.get_shape(), inputs.get_shape())
            return norm / (1 + norm * norm) * inputs

    @staticmethod
    def squash(vector):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A 5-D tensor with shape [batch_size, 1, num_caps, vec_len, 1],
        Returns:
            A 5-D tensor with the same shape as vector but squashed in 4rd and 5th dimensions.
        '''

        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
        vec_norm = tf.sqrt(vec_squared_norm)
        scalar_factor = vec_norm / (1 + vec_squared_norm)
        vec_squashed = scalar_factor * vector  # element-wise
        return vec_squashed

    def update_b(self, U_hat, V):
        with tf.variable_scope('Update_B'):
            V = tf.expand_dims(V, axis=1)
            V = tf.tile(V, multiples=[1, self.previous_num_capsule, 1, 1])
            mult = tf.multiply(U_hat, V)
            ret = tf.reduce_sum(mult, axis=3)
            return ret


def capsule(inputs,
            num_capsule,
            vec_len,
            output_type='scalar',
            weight_initializer=init_ops.random_normal_initializer(),
            activity_regularizer=None,
            trainable=True,
            name=None,
            reuse=None):
    layer = CapsuleLayer(num_capsule=num_capsule,
                         vec_len=vec_len,
                         output_type=output_type,
                         weight_initializer=weight_initializer,
                         activity_regularizer=activity_regularizer,
                         trainable=trainable,
                         name=name,
                         dtype=inputs.dtype.base_dtype,
                         _scope=name,
                         _reuse=reuse)
    return layer.apply(inputs)
