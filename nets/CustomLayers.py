# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.layers import (Layer,
                                     Conv2D,
                                     BatchNormalization)
from tensorflow.python.keras.utils import conv_utils


class Padding2D(Layer):
    def __init__(self,
                 pad_size: int or tuple=None,
                 data_format: str="channels_last",
                 **kwargs):
        super(Padding2D, self).__init__(**kwargs)
        assert data_format in ["channels_first", "channels_last"]
        self.pad_size = pad_size
        self.data_format = data_format
        if self.data_format == 'channels_first':
            if isinstance(pad_size, int):
                self.padding = [[0, 0],
                                [0, 0],
                                [pad_size, ] * 2,
                                [pad_size, ] * 2]
            else:
                self.padding = [[0, 0],
                                [0, 0],
                                [pad_size[0], pad_size[1]],
                                [pad_size[0], pad_size[1]]]
        else:
            if isinstance(pad_size, int):
                self.padding = [[0, 0],
                                [pad_size, ] * 2,
                                [pad_size, ] * 2,
                                [0, 0]]
            else:
                self.padding = [[0, 0],
                                [pad_size[0], pad_size[1]],
                                [pad_size[0], pad_size[1]],
                                [0, 0]]

    def get_config(self):
        config = super(Padding2D, self).get_config()
        config.update({
            'pad_size': self.pad_size,
            'data_format': self.data_format
        })
        return config

    def call(self, input, padding_mode='CONSTANT', **kwargs):
        assert padding_mode in ['CONSTANT', 'REFLECT', 'SYMMETRIC']

        return tf.pad(input, self.padding, mode=padding_mode)


class ShuffleUnit(Layer):
    """
    Feature shuffling
    """
    def __init__(self,
                 params,
                 **kwargs):
        super(ShuffleUnit, self).__init__(**kwargs)
        self.params = params

    def get_config(self):

        raise Exception("关注并联系作者获取通道打乱机制, e-mail:m13541280433@163.com")

    def call(self, input, *args, **kwargs):

        raise Exception("关注并联系作者获取通道打乱机制, e-mail:m13541280433@163.com")


class GroupConv2D(Conv2D):
    def __init__(self,
                 groups: int=None,
                 **kwargs):
        super(GroupConv2D, self).__init__(**kwargs)
        self.groups = groups
        self.padding = self.padding.upper()
        self.data_format = conv_utils.convert_data_format(self.data_format, ndim=4)

    def get_config(self):

        config = super(GroupConv2D, self).get_config()
        config.update({
            'groups': self.groups
        })

        return config

    def build(self, input_shape):
        if self.data_format == 'NCHW':
            channel_axis = 1
        else:
            channel_axis = -1

        if not input_shape[channel_axis]:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, ) * 2

        kernel_shape = self.kernel_size + (input_shape[channel_axis]//self.groups, self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      trainable=True,
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        trainable=True,
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.built = True

    def call(self, input):

        inputs = tf.split(input, num_or_size_splits=self.groups,
                          axis=1 if self.data_format == 'NCHW' else -1)
        kernels = tf.split(self.kernel, num_or_size_splits=self.groups,
                           axis=1 if self.data_format == 'NCHW' else -1)

        feats = [tf.nn.conv2d(input, filters=kernels[i], strides=self.strides,
                                padding=self.padding, data_format=self.data_format)
                   for i, input in enumerate(inputs)]

        output = tf.concat(feats, axis=1 if self.data_format == 'NCHW' else -1)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias, data_format=self.data_format)

        if self.activation:
            output = self.activation(output)

        return output
