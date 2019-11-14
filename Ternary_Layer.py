import tensorflow as tf
import numpy as np
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import activations


#STE implemented quantizer
@tf.custom_gradient
def tfquantizerCG (w):
    threshold = 0.7 * tf.reduce_sum(tf.abs(w)) / tf.cast(tf.size(w), w.dtype)
    out = tf.sign(tf.sign(w + threshold) + tf.sign(w - threshold))

    def grad(dy):
        g = tf.cond(tf.reduce_mean(tf.abs(out)) > 0, true_fn=lambda: dy, false_fn=lambda: 0*dy)
        return g

    return tf.sign(tf.sign(w + threshold) + tf.sign(w - threshold)), grad


#The quantizer without STE
def tfquantizer (w):
    threshold = 0.7 * tf.reduce_sum(tf.abs(w)) / tf.cast(tf.size(w), w.dtype)
    return tf.sign(tf.sign(w + threshold) + tf.sign(w - threshold))

class Weights (tf.keras.constraints.Constraint):
    def __init__(self, clip_value=1):
        self.clip_value = clip_value


    def __call__(self, w):
        return tf.clip_by_value(w, -self.clip_value, self.clip_value)

    def get_config(self):
        return {"clip_value": self.clip_value}

class MyLayer(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.output_dim = units
        super(MyLayer, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape):
        self.kernel = self.add_weight(
        'kernel',
        shape=[self.units],
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        dtype=self.dtype,
        trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        ternary_weights = tfquantizer(self.kernel)

        #Turning the vector of the weights into Diagonal Matirces
        ternary_weights_matrix =tf.diag(ternary_weights)

        if self.use_bias:
            return tf.add(self.bias, tf.matmul(x, ternary_weights_matrix))
        else:
            return tf.matmul(x, ternary_weights_matrix)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))