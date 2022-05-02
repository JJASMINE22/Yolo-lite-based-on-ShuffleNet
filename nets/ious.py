# -*- coding: UTF-8 -*-
'''
@Project ：Yolo-lite-based-on-ShuffleNet
@File    ：ious.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import math
import tensorflow as tf


def box_ciou(b1, b2):
    """
    :param b1: shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :param b2: shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    :return: ciou, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #-----------------------------------------------------------#
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    #-----------------------------------------------------------#
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    #-----------------------------------------------------------#
    #   iou         (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / tf.maximum(union_area, 1e-7) # if no frame exists, avoid area of 0

    #-----------------------------------------------------------#
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    center_distance = tf.reduce_sum(tf.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = tf.minimum(b1_mins, b2_mins)
    enclose_maxes = tf.maximum(b1_maxes, b2_maxes)
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.0)
    #-----------------------------------------------------------#
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / tf.maximum(enclose_diagonal, 1e-7)
    
    v = 4 * tf.square(tf.math.atan2(b1_wh[..., 0], tf.maximum(b1_wh[..., 1], 1e-7)) -
                      tf.math.atan2(b2_wh[..., 0], tf.maximum(b2_wh[..., 1], 1e-7))) / (math.pi * math.pi)
    alpha = v / tf.maximum((1.0 - iou + v), 1e-7)
    ciou = ciou - alpha * v

    ciou = ciou[..., tf.newaxis]
    ciou = tf.where(tf.math.is_nan(ciou), tf.zeros_like(ciou), ciou)
    return ciou
