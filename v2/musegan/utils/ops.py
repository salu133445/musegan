"""Operations for implementing binary neurons. Code is from the R2RT blog post:
https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html (slightly adapted)
"""
import tensorflow as tf
from tensorflow.python.framework import ops

def binary_round(x):
    """
    Rounds a tensor whose values are in [0,1] to a tensor with values in
    {0, 1}, using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryRound") as name:
        with g.gradient_override_map({"Round": "Identity"}):
            return tf.round(x, name=name)

def bernoulli_sample(x):
    """
    Uses a tensor whose values are in [0,1] to sample a tensor with values
    in {0, 1}, using the straight through estimator for the gradient.

    E.g., if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6,
    and 0 otherwise, and the gradient will be pass-through (identity).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BernoulliSample") as name:
        with g.gradient_override_map({"Ceil": "Identity",
                                      "Sub": "BernoulliSample_ST"}):
            return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

@ops.RegisterGradient("BernoulliSample_ST")
def bernoulli_sample_ST(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]

def pass_through_sigmoid(x, slope=1):
    """Sigmoid that uses identity function as its gradient"""
    g = tf.get_default_graph()
    with ops.name_scope("PassThroughSigmoid") as name:
        with g.gradient_override_map({"Sigmoid": "Identity"}):
            return tf.sigmoid(x, name=name)

def binary_stochastic_ST(x, slope_tensor=None, pass_through=True,
                        stochastic=True):
    """
    Sigmoid followed by either a random sample from a bernoulli distribution
    according to the result (binary stochastic neuron) (default), or a
    sigmoid followed by a binary step function (if stochastic == False).
    Uses the straight through estimator. See
    https://arxiv.org/abs/1308.3432.

    Arguments:
    * x: the pre-activation / logit tensor
    * slope_tensor: if passThrough==False, slope adjusts the slope of the
        sigmoid function for purposes of the Slope Annealing Trick (see
        http://arxiv.org/abs/1609.01704)
    * pass_through: if True (default), gradient of the entire function is 1
        or 0; if False, gradient of 1 is scaled by the gradient of the
        sigmoid (required if Slope Annealing Trick is used)
    * stochastic: binary stochastic neuron if True (default), or step
        function if False
    """
    if slope_tensor is None:
        slope_tensor = tf.constant(1.0)

    if pass_through:
        p = pass_through_sigmoid(x)
    else:
        p = tf.sigmoid(slope_tensor * x)

    if stochastic:
        return bernoulli_sample(p), p
    else:
        return binary_round(p), p

def binary_stochastic_REINFORCE(x, loss_op_name="loss_by_example"):
    """
    Sigmoid followed by a random sample from a bernoulli distribution
    according to the result (binary stochastic neuron). Uses the REINFORCE
    estimator. See https://arxiv.org/abs/1308.3432.

    NOTE: Requires a loss operation with name matching the argument for
    loss_op_name in the graph. This loss operation should be broken out by
    example (i.e., not a single number for the entire batch).
    """
    g = tf.get_default_graph()

    with ops.name_scope("BinaryStochasticREINFORCE"):
        with g.gradient_override_map({"Sigmoid": "BinaryStochastic_REINFORCE",
                                      "Ceil": "Identity"}):
            p = tf.sigmoid(x)

            reinforce_collection = g.get_collection("REINFORCE")
            if not reinforce_collection:
                g.add_to_collection("REINFORCE", {})
                reinforce_collection = g.get_collection("REINFORCE")
            reinforce_collection[0][p.op.name] = loss_op_name

            return tf.ceil(p - tf.random_uniform(tf.shape(x)))


@ops.RegisterGradient("BinaryStochastic_REINFORCE")
def _binaryStochastic_REINFORCE(op, _):
    """Unbiased estimator for binary stochastic function based on REINFORCE."""
    loss_op_name = op.graph.get_collection("REINFORCE")[0][op.name]
    loss_tensor = op.graph.get_operation_by_name(loss_op_name).outputs[0]

    sub_tensor = op.outputs[0].consumers()[0].outputs[0] #subtraction tensor
    ceil_tensor = sub_tensor.consumers()[0].outputs[0] #ceiling tensor

    outcome_diff = (ceil_tensor - op.outputs[0])

    # Provides an early out if we want to avoid variance adjustment for
    # whatever reason (e.g., to show that variance adjustment helps)
    if op.graph.get_collection("REINFORCE")[0].get("no_variance_adj"):
        return outcome_diff * tf.expand_dims(loss_tensor, 1)

    outcome_diff_sq = tf.square(outcome_diff)
    outcome_diff_sq_r = tf.reduce_mean(outcome_diff_sq, reduction_indices=0)
    outcome_diff_sq_loss_r = tf.reduce_mean(
        outcome_diff_sq * tf.expand_dims(loss_tensor, 1), reduction_indices=0)

    l_bar_num = tf.Variable(tf.zeros(outcome_diff_sq_r.get_shape()),
                            trainable=False)
    l_bar_den = tf.Variable(tf.ones(outcome_diff_sq_r.get_shape()),
                            trainable=False)

    # Note: we already get a decent estimate of the average from the minibatch
    decay = 0.95
    train_l_bar_num = tf.assign(l_bar_num, l_bar_num*decay +\
                                            outcome_diff_sq_loss_r*(1-decay))
    train_l_bar_den = tf.assign(l_bar_den, l_bar_den*decay +\
                                            outcome_diff_sq_r*(1-decay))


    with tf.control_dependencies([train_l_bar_num, train_l_bar_den]):
        l_bar = train_l_bar_num/(train_l_bar_den + 1e-4)
        l = tf.tile(tf.expand_dims(loss_tensor, 1),
                    tf.constant([1, l_bar.get_shape().as_list()[0]]))
        return outcome_diff * (l - l_bar)

def binary_wrapper(pre_activations_tensor, estimator,
                   stochastic_tensor=tf.constant(True), pass_through=True,
                   slope_tensor=tf.constant(1.0)):
    """
    Turns a layer of pre-activations (logits) into a layer of binary
    stochastic neurons

    Keyword arguments:
    *estimator: either ST or REINFORCE
    *stochastic_tensor: a boolean tensor indicating whether to sample from a
        bernoulli distribution (True, default) or use a step_function (e.g.,
        for inference)
    *pass_through: for ST only - boolean as to whether to substitute
        identity derivative on the backprop (True, default), or whether to
        use the derivative of the sigmoid
    *slope_tensor: for ST only - tensor specifying the slope for purposes of
        slope annealing trick
    """
    if estimator == 'straight_through':
        if pass_through:
            return tf.cond(
                stochastic_tensor,
                lambda: binary_stochastic_ST(pre_activations_tensor),
                lambda: binary_stochastic_ST(pre_activations_tensor,
                                             stochastic=False))
        else:
            return tf.cond(
                stochastic_tensor,
                lambda: binary_stochastic_ST(pre_activations_tensor,
                                             slope_tensor, False),
                lambda: binary_stochastic_ST(pre_activations_tensor,
                                             slope_tensor, False, False))

    elif estimator == 'reinforce':
        # binaryStochastic_REINFORCE was designed to only be stochastic, so
        # using the ST version for the step fn for purposes of using step
        # fn at evaluation / not for training
        return tf.cond(
            stochastic_tensor,
            lambda: binary_stochastic_REINFORCE(pre_activations_tensor),
            lambda: binary_stochastic_ST(pre_activations_tensor,
                                         stochastic=False))

    else:
        raise ValueError("Unrecognized estimator.")
