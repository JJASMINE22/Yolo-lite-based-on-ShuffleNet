# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,
                                     DepthwiseConv2D,
                                     Concatenate,
                                     MaxPooling2D,
                                     BatchNormalization,
                                     ZeroPadding2D,
                                     Activation,
                                     LeakyReLU)
from tensorflow.keras import initializers, regularizers
from nets.CustomLayers import GroupConv2D, ShuffleUnit, Padding2D


def grouped_res_block(init, expansion: int or float, target_size: int, stride:int or tuple, groups: int):
    """
    init → pointwise group conv → depthwise conv → point wise conv → concat or add
    """
    input_size = init.shape[-1]
    expand_size = int(input_size * expansion)

    x = GroupConv2D(filters=expand_size,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=False,
                    groups=groups,
                    kernel_initializer=initializers.random_normal,
                    kernel_regularizer=regularizers.l2(5e-4))(init)
    x = BatchNormalization(epsilon=1e-5, momentum=0.999)(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = ShuffleUnit(groups=groups)(x)

    x = Padding2D(pad_size=(1, 1))(x, padding_mode='REFLECT')
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=stride,
                        padding='VALID',
                        use_bias=False,
                        kernel_initializer=initializers.random_normal,
                        depthwise_initializer=initializers.random_normal,
                        kernel_regularizer=regularizers.l2(5e-4),
                        depthwise_regularizer=regularizers.l2(5e-4))(x)

    x = BatchNormalization(epsilon=1e-5, momentum=0.999)(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = GroupConv2D(filters=target_size,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    use_bias=False,
                    groups=groups,
                    kernel_initializer=initializers.random_normal,
                    kernel_regularizer=regularizers.l2(5e-4))(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.999)(x)

    if input_size != target_size:

        init = Padding2D(pad_size=(1, 1))(init, padding_mode='REFLECT')
        init = MaxPooling2D(pool_size=(3, 3),
                            strides=stride,
                            padding='VALID')(init)
        x = Concatenate(axis=-1)([init, x])
    else:
        x = x + init

    output = LeakyReLU(alpha=0.3)(x)

    return output

def ShuffleNet(init):
    x = Padding2D(pad_size=(1, 1))(init, padding_mode='REFLECT')

    # 416,416,3 -> 208,208,16
    x = Conv2D(filters=16,
               kernel_size=(3, 3),
               strides=(2, 2),
               padding='VALID',
               use_bias=False,
               kernel_initializer=initializers.random_normal,
               kernel_regularizer=regularizers.l2(5e-4))(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.999)(x)
    x = LeakyReLU(alpha=0.3)(x)

    # 208,208,16 -> 208,208,16
    x = grouped_res_block(x, expansion=1, target_size=16, stride=(1, 1), groups=1)

    # 208,208,16 -> 104,104,40
    x = grouped_res_block(x, expansion=2, target_size=24, stride=(2, 2), groups=1)
    x = grouped_res_block(x, expansion=2, target_size=40, stride=(1, 1), groups=1)

    # 208,208,40 -> 52,52,96
    x = grouped_res_block(x, expansion=2, target_size=56, stride=(2, 2), groups=2)
    x = grouped_res_block(x, expansion=2, target_size=96, stride=(1, 1), groups=2)
    x = grouped_res_block(x, expansion=2, target_size=96, stride=(1, 1), groups=2)
    feat1 = x

    # 208,208,96 -> 26,26,224
    x = grouped_res_block(x, expansion=3, target_size=128, stride=(2, 2), groups=4)
    x = grouped_res_block(x, expansion=3, target_size=224, stride=(1, 1), groups=4)
    x = grouped_res_block(x, expansion=3, target_size=224, stride=(1, 1), groups=4)
    x = grouped_res_block(x, expansion=3, target_size=224, stride=(1, 1), groups=4)
    x = grouped_res_block(x, expansion=3, target_size=224, stride=(1, 1), groups=4)
    x = grouped_res_block(x, expansion=3, target_size=224, stride=(1, 1), groups=4)
    feat2 = x

    # 208,208,224 -> 13,13,512
    x = grouped_res_block(x, expansion=3, target_size=288, stride=(2, 2), groups=8)
    x = grouped_res_block(x, expansion=3, target_size=512, stride=(1, 1), groups=8)
    x = grouped_res_block(x, expansion=3, target_size=512, stride=(1, 1), groups=8)
    feat3 = x

    return feat1, feat2, feat3
