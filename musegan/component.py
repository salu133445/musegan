"""Base class for the components.
"""
from collections import OrderedDict
import tensorflow as tf
from musegan.utils.neuralnet import NeuralNet

class Component(object):
    """Base class for components."""
    def __init__(self, tensor_in, condition, slope_tensor=None):
        if not isinstance(tensor_in, (tf.Tensor, list, dict)):
            raise TypeError("`tensor_in` must be of tf.Tensor type or a list "
                            "(or dict) of tf.Tensor objects")
        if isinstance(tensor_in, list):
            for tensor in tensor_in:
                if not isinstance(tensor, tf.Tensor):
                    raise TypeError("`tensor_in` must be of tf.Tensor type or "
                                    "a list (or dict) of tf.Tensor objects")
        if isinstance(tensor_in, dict):
            for key in tensor_in:
                if not isinstance(tensor_in[key], tf.Tensor):
                    raise TypeError("`tensor_in` must be of tf.Tensor type or "
                                    "a list (or dict) of tf.Tensor objects")

        self.tensor_in = tensor_in
        self.condition = condition
        self.slope_tensor = slope_tensor

        self.scope = None
        self.tensor_out = tensor_in
        self.nets = OrderedDict()
        self.vars = None

    def __repr__(self):
        if isinstance(self.tensor_in, tf.Tensor):
            input_shape = self.tensor_in.get_shape()
        else:
            input_shape = ', '.join([
                '{}: {}'.format(key, self.tensor_in[key].get_shape())
                for key in self.tensor_in])
        return "Component({}, input_shape={}, output_shape={})".format(
            self.scope.name, input_shape, str(self.tensor_out.get_shape()))

    def get_summary(self):
        """Return the summary string."""
        cleansed_nets = []
        for net in self.nets.values():
            if isinstance(net, NeuralNet):
                if net.scope is not None:
                    cleansed_nets.append(net)
            if isinstance(net, list):
                if net[0].scope is not None:
                    cleansed_nets.append(net[0])

        if isinstance(self.tensor_in, tf.Tensor):
            input_strs = ["{:50}{}".format('Input', self.tensor_in.get_shape())]
        else:
            input_strs = ["{:50}{}".format('Input - ' + key,
                                           self.tensor_in[key].get_shape())
                          for key in self.tensor_in]

        return '\n'.join(
            ["{:-^80}".format(' ' + self.scope.name + ' ')] + input_strs
            + ['-' * 80 + '\n' + x.get_summary() for x in cleansed_nets]
        )
