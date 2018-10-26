"""Classes for neural networks and layers.
"""
import numpy as np
import tensorflow as tf
from musegan.utils.ops import binary_stochastic_ST

SUPPORTED_LAYER_TYPES = (
    'reshape', 'mean', 'sum', 'dense', 'identity', 'conv1d', 'conv2d', 'conv3d',
    'transconv2d', 'transconv3d', 'avgpool2d', 'avgpool3d', 'maxpool2d',
    'maxpool3d'
)

class Layer(object):
    """Base class for layers."""
    def __init__(self, tensor_in, structure=None, condition=None,
                 slope_tensor=None, name=None, reuse=None):
        if not isinstance(tensor_in, tf.Tensor):
            raise TypeError("`tensor_in` must be of tf.Tensor type")

        self.tensor_in = tensor_in

        if structure is not None:
            with tf.variable_scope(name, reuse=reuse) as scope:
                self.scope = scope
                if structure[0] not in SUPPORTED_LAYER_TYPES:
                    raise ValueError("Unknown layer type at " + self.scope.name)
                self.layer_type = structure[0]
                self.tensor_out = self.build(structure, condition, slope_tensor)
                self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.scope.name)
        else:
            self.scope = None
            self.layer_type = 'bypass'
            self.tensor_out = tensor_in
            self.vars = []

    def __repr__(self):
        return "Layer({}, type={}, input_shape={}, output_shape={})".format(
            self.scope.name, self.layer_type, self.tensor_in.get_shape(),
            self.tensor_out.get_shape())

    def get_summary(self):
        """Return the summary string."""
        return "{:36} {:12} {:30}".format(
            self.scope.name, self.layer_type, str(self.tensor_out.get_shape()))

    def build(self, structure, condition, slope_tensor):
        """Build the layer."""
        # Mean layers
        if self.layer_type == 'mean':
            keepdims = structure[2] if len(structure) > 2 else None
            return tf.reduce_mean(self.tensor_in, structure[1], keepdims,
                                  name='mean')

        # Summation layers
        if self.layer_type == 'sum':
            keepdims = structure[2] if len(structure) > 2 else None
            return tf.reduce_sum(self.tensor_in, structure[1], keepdims,
                                 name='sum')

        # Reshape layers
        if self.layer_type == 'reshape':
            if np.prod(structure[1]) != np.prod(self.tensor_in.get_shape()[1:]):
                raise ValueError("Bad reshape size: {} to {} at {}".format(
                    self.tensor_in.get_shape()[1:], structure[1],
                    self.scope.name))
            if isinstance(structure[1], int):
                reshape_shape = (-1, structure[1])
            else:
                reshape_shape = (-1,) + structure[1]
            return tf.reshape(self.tensor_in, reshape_shape, 'reshape')

        # Pooling layers
        if self.layer_type == 'avgpool2d':
            return tf.layers.average_pooling2d(self.tensor_in, structure[1][0],
                                               structure[1][1],
                                               name='avgpool2d')
        if self.layer_type == 'maxpool2d':
            return tf.layers.max_pooling2d(self.tensor_in, structure[1][0],
                                           structure[1][1], name='maxpool2d')
        if self.layer_type == 'avgpool3d':
            return tf.layers.average_pooling3d(self.tensor_in, structure[1][0],
                                               structure[1][1],
                                               name='avgpool3d')
        if self.layer_type == 'maxpool3d':
            return tf.layers.max_pooling3d(self.tensor_in, structure[1][0],
                                           structure[1][1], name='maxpool3d')

        # Condition
        if condition is None:
            self.conditioned = self.tensor_in
        elif self.layer_type == 'dense':
            self.conditioned = tf.concat([self.tensor_in, condition], 1)
        elif self.layer_type in ('conv1d', 'conv2d', 'transconv2d', 'conv3d',
                                 'transconv3d'):
            if self.layer_type == 'conv1d':
                reshape_shape = (-1, 1, condition.get_shape()[1])
            elif self.layer_type in ('conv2d', 'transconv2d'):
                reshape_shape = (-1, 1, 1, condition.get_shape()[1])
            else: # ('conv3d', 'transconv3d')
                reshape_shape = (-1, 1, 1, 1, condition.get_shape()[1])
            reshaped = tf.reshape(condition, reshape_shape)
            out_shape = ([-1] + self.tensor_in.get_shape()[1:-1]
                         + [condition.get_shape()[1]])
            to_concat = reshaped * tf.ones(out_shape)
            self.conditioned = tf.concat([self.tensor_in, to_concat], -1)

        # Core layers (dense, convolutional or identity layer)
        if self.layer_type == 'dense':
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)
            self.core = tf.layers.dense(self.conditioned, structure[1],
                                        kernel_initializer=kernel_initializer,
                                        name='dense')
        elif self.layer_type == 'identity':
            self.core = self.conditioned
        else:
            filters = structure[1][0]
            kernel_size = structure[1][1]
            strides = structure[1][2] if len(structure[1]) > 2 else 1
            padding = structure[1][3] if len(structure[1]) > 3 else 'valid'
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.02)

            if self.layer_type == 'conv1d':
                self.core = tf.layers.conv1d(
                    self.conditioned, filters, kernel_size, strides, padding,
                    kernel_initializer=kernel_initializer, name='conv1d')
            elif self.layer_type == 'conv2d':
                self.core = tf.layers.conv2d(
                    self.conditioned, filters, kernel_size, strides, padding,
                    kernel_initializer=kernel_initializer, name='conv2d')
            elif self.layer_type == 'transconv2d':
                self.core = tf.layers.conv2d_transpose(
                    self.conditioned, filters, kernel_size, strides, padding,
                    kernel_initializer=kernel_initializer, name='transconv2d')
            elif self.layer_type == 'conv3d':
                self.core = tf.layers.conv3d(
                    self.conditioned, filters, kernel_size, strides, padding,
                    kernel_initializer=kernel_initializer, name='conv3d')
            elif self.layer_type == 'transconv3d':
                self.core = tf.layers.conv3d_transpose(
                    self.conditioned, filters, kernel_size, strides, padding,
                    kernel_initializer=kernel_initializer, name='transconv3d')

        # normalization layer
        if len(structure) > 2:
            if structure[2] not in (None, 'bn', 'in', 'ln'):
                raise ValueError("Unknown normalization at " +  self.scope.name)
            normalization = structure[2]
        else:
            normalization = None

        if normalization is None:
            self.normalized = self.core
        elif normalization == 'bn':
            self.normalized = tf.layers.batch_normalization(
                self.core, name='batch_norm')
        elif normalization == 'in':
            self.normalized = tf.contrib.layers.instance_norm(
                self.core, scope='instance_norm')
        elif normalization == 'ln':
            self.normalized = tf.contrib.layers.layer_norm(
                self.core, scope='layer_norm')

        # activation
        if len(structure) > 3:
            if structure[3] not in (None, 'tanh', 'sigmoid', 'relu', 'lrelu',
                                    'bernoulli', 'round'):
                raise ValueError("Unknown activation at " + self.scope.name)
            activation = structure[3]
        else:
            activation = None

        if activation is None:
            self.activated = self.normalized
        elif activation == 'tanh':
            self.activated = tf.nn.tanh(self.normalized, 'tanh')
        elif activation == 'sigmoid':
            self.activated = tf.nn.sigmoid(self.normalized, 'sigmoid')
        elif activation == 'relu':
            self.activated = tf.nn.relu(self.normalized, 'relu')
        elif activation == 'lrelu':
            self.activated = tf.nn.leaky_relu(self.normalized, name='lrelu')
        elif activation == 'bernoulli':
            self.activated, self.preactivated = binary_stochastic_ST(
                self.normalized, slope_tensor, False, True)
        elif activation == 'round':
            self.activated, self.preactivated = binary_stochastic_ST(
                self.normalized, slope_tensor, False, False)

        return self.activated

class NeuralNet(object):
    """Base class for neural networks."""
    def __init__(self, tensor_in, architecture=None, condition=None,
                 slope_tensor=None, name='NeuralNet', reuse=None):
        if not isinstance(tensor_in, tf.Tensor):
            raise TypeError("`tensor_in` must be of tf.Tensor type")

        self.tensor_in = tensor_in
        self.condition = condition
        self.slope_tensor = slope_tensor

        if architecture is not None:
            with tf.variable_scope(name, reuse=reuse) as scope:
                self.scope = scope
                self.layers = self.build(architecture)
                self.tensor_out = self.layers[-1].tensor_out
                self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              self.scope.name)
        else:
            self.scope = None
            self.layers = []
            self.tensor_out = tensor_in
            self.vars = []

    def __repr__(self):
        return "NeuralNet({}, input_shape={}, output_shape={})".format(
            self.scope.name, self.tensor_in.get_shape(),
            self.tensor_out.get_shape())

    def get_summary(self):
        """Return the summary string."""
        return '\n'.join(
            ['[{}]'.format(self.scope.name),
             "{:49} {}".format('Input', self.tensor_in.get_shape())]
            + [x.get_summary() for x in self.layers])

    def build(self, architecture):
        """Build the neural network."""
        layers = []
        for idx, structure in enumerate(architecture):
            if idx > 0:
                prev_layer = layers[idx-1].tensor_out
            else:
                prev_layer = self.tensor_in

            # Skip connections
            if len(structure) > 4:
                skip_connection = structure[4][0]
            else:
                skip_connection = None

            if skip_connection is None:
                connected = prev_layer
            elif skip_connection == 'add':
                connected = prev_layer + layers[structure[4][1]].tensor_out
            elif skip_connection == 'concat':
                connected = tf.concat(
                    [prev_layer, layers[structure[4][1]].tensor_out], -1)

            # Build layer
            layers.append(Layer(connected, structure,
                          slope_tensor=self.slope_tensor,
                          name='Layer_{}'.format(idx)))
        return layers
