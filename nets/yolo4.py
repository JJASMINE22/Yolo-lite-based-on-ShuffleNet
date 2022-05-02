import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input,
                                     Conv2D,
                                     Activation,
                                     Concatenate,
                                     DepthwiseConv2D,
                                     MaxPooling2D,
                                     UpSampling2D,
                                     BatchNormalization,
                                     LeakyReLU)
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Model
from nets.shufflenet import ShuffleNet

#---------------------------------------------------#
#   Conv2D + BatchNormalization + LeakyRelu
#---------------------------------------------------#
def conv_block(init, target_size, kernel_size, **kwargs):

    kwargs['padding'] = 'VALID' if kwargs.get('strides') == (2, 2) else 'SAME'
    kwargs.update({'kernel_initializer':initializers.random_normal,
                   'kernel_regularizer':regularizers.l2(5e-4),
                   'use_bias': False})

    x = Conv2D(target_size, kernel_size, **kwargs)(init)
    x = BatchNormalization()(x)

    output = LeakyReLU(alpha=0.3)(x)

    return output

#---------------------------------------------------#
#   DepthwiseConv2D + BatchNormalization + LeakyRelu
#---------------------------------------------------#
def depthwise_conv_block(inputs, pointwise_filters, alpha=1,
                          depth_multiplier=1, strides=(1, 1)):

    pointwise_filters = int(pointwise_filters * alpha)
    
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        padding='SAME',
                        kernel_initializer=initializers.random_normal,
                        kernel_regularizer=regularizers.l2(5e-4),
                        depth_multiplier=depth_multiplier,
                        use_bias=False)(inputs)

    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv2D(filters=pointwise_filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='SAME',
               kernel_initializer=initializers.random_normal,
               kernel_regularizer=regularizers.l2(5e-4),
               use_bias=False)(x)
    x = BatchNormalization()(x)

    output = LeakyReLU(alpha=0.3)(x)

    return output


def make_five_convs(x, num_filters):
    x = conv_block(x, num_filters, (1, 1))
    x = depthwise_conv_block(x, num_filters*2, alpha=1)
    x = conv_block(x, num_filters, (1, 1))
    x = depthwise_conv_block(x, num_filters*2, alpha=1)
    x = conv_block(x, num_filters, (1, 1))
    return x

#   Panet
def yolo_body(input_shape, num_anchors, num_classes, backbone="shufflenet", alpha=1):
    init = Input(shape=input_shape)

    if backbone == "shufflenet":
        feat1, feat2, feat3 = ShuffleNet(init)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use groupnet.'.format(backbone))
    
    P5 = conv_block(feat3, int(512 * alpha), (1, 1))
    P5 = depthwise_conv_block(P5, int(1024 * alpha))
    P5 = conv_block(P5, int(512 * alpha), (1, 1))
    maxpool1 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = conv_block(P5, int(512 * alpha), (1, 1))
    P5 = depthwise_conv_block(P5, int(1024 * alpha))
    P5 = conv_block(P5, int(512 * alpha), (1, 1))

    P5 = conv_block(P5, int(256 * alpha), (1, 1))
    P5_upsample = UpSampling2D(size=(2, 2))(P5)

    P4 = conv_block(feat2, int(256 * alpha), (1, 1))
    P4 = Concatenate()([P4, P5_upsample])
    P4 = make_five_convs(P4, int(256 * alpha))

    P4 = conv_block(P4, int(128 * alpha), (1, 1))
    P4_upsample = UpSampling2D(size=(2, 2))(P4)

    P3 = conv_block(feat1, int(128 * alpha), (1, 1))
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_five_convs(P3, int(128 * alpha))

    #---------------------------------------------------#
    #   y3  (batch_size,52,52,3,5+num_classes)
    #---------------------------------------------------#
    P3_output = depthwise_conv_block(P3, int(256 * alpha))
    P3_output = Conv2D(num_anchors*(num_classes+5), (1, 1), padding='SAME',
                       kernel_initializer=initializers.random_normal,
                       kernel_regularizer=regularizers.l2(5e-4))(P3_output)

    P3_downsample = depthwise_conv_block(P3, int(256 * alpha), strides=(2, 2))
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4, int(256 * alpha))

    #---------------------------------------------------#
    #   y2  (batch_size,26,26,3,5+num_classes)
    #---------------------------------------------------#
    P4_output = depthwise_conv_block(P4, int(512 * alpha))
    P4_output = Conv2D(num_anchors*(num_classes+5), (1, 1), padding='SAME',
                       kernel_initializer=initializers.random_normal,
                       kernel_regularizer=regularizers.l2(5e-4))(P4_output)

    P4_downsample = depthwise_conv_block(P4, int(512 * alpha), strides=(2, 2))
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5, int(512 * alpha))

    #---------------------------------------------------#
    #   y1  (batch_size,13,13,3,5+num_classes)
    #---------------------------------------------------#
    P5_output = depthwise_conv_block(P5, int(1024 * alpha))
    P5_output = Conv2D(num_anchors*(num_classes+5), (1, 1), padding='SAME',
                       kernel_initializer=initializers.random_normal,
                       kernel_regularizer=regularizers.l2(5e-4))(P5_output)

    return Model(init, [P5_output, P4_output, P3_output])


def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    anchors_tensor = tf.reshape(tf.cast(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])

    #---------------------------------------------------#
    #   construct grid of different sizes of sensory fields
    #   (13, 13, 1, 2)
    #---------------------------------------------------#
    grid_shape = tf.shape(feats)[1:3]
    grids = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))
    grid_xy = tf.stack(grids, axis=-1)
    grid = grid_xy[..., tf.newaxis, :]
    grid = tf.cast(grid, dtype=feats.dtype)

    #---------------------------------------------------#
    #  get feats (batch_size,13,13,3,5+num_classes)
    #---------------------------------------------------#
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #---------------------------------------------------#
    #  get box_xy, box_wh
    #---------------------------------------------------#
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[..., ::-1], dtype=feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], dtype=feats.dtype)

    raise Exception("置信度与分类概率维度的激活方式有变, 关注并联系作者获取, e-mail:m13541280433@163.com")


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    input_shape = tf.cast(input_shape, dtype=box_yx.dtype)
    image_shape = tf.cast(image_shape, dtype=box_yx.dtype)

    new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
    #-----------------------------------------------------------------#
    # restore the state of the image before offset
    # check the function of get_random_data
    #-----------------------------------------------------------------#
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    boxes *= tf.concat([image_shape, image_shape], axis=-1)
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   box_xy (-1, field, field, 3, 2)
    #   box_wh (-1, field, field, 3, 2)
    #   box_confidence (-1, field, field, 3, 1);
    #   box_class_probs (-1, field, field, 3, class_num)
    #-----------------------------------------------------------------#
    box_xy, box_wh, box_confidences, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)

    if letterbox_image:
        #  remove gray bars from images, and resize
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        # resize
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = tf.cast(input_shape, dtype=box_yx.dtype)
        image_shape = tf.cast(image_shape, dtype=box_yx.dtype)

        boxes = tf.concat([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ], axis=-1)
    # results of each sample from the batch
    boxes = tf.reshape(boxes, [-1, 4])
    box_confidences = tf.squeeze(box_confidences, axis=-1)
    box_confidences = tf.reshape(box_confidences, [-1])
    box_class_probs = tf.reshape(box_class_probs, [-1, num_classes])
    return boxes, box_confidences, box_class_probs


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=50,
              score_threshold=.5,
              iou_threshold=.5,
              letterbox_image=False):
    num_layers = len(yolo_outputs)
    #-----------------------------------------------------------#
    #   13x13 anchor [142, 110], [192, 243], [459, 401]
    #   26x26 anchor [36, 75], [76, 55], [72, 146]
    #   52x52 anchor [12, 16], [19, 36], [40, 28]
    #   also you can generate anchors by activate kmeans_for_anchors.py
    #-----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    batch_size = tf.shape(yolo_outputs[0])[0]
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32

    raise Exception("支持批量预测, 目标检出规则有变, 关注并联系作者获取, e-mail:m13541280433@163.com")
