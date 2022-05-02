import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss
from nets.ious import box_ciou

def smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(tf.shape(y_true)[-1], dtype=tf.float32)
    label_smoothing = tf.cast(label_smoothing, dtype=tf.float32)
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


class EagerMaskedReducalNLL(Loss):
    """
    calculate yolo classification error based on NLL
    by using object mask
    """
    def __init__(self,
                 params,
                 **kwargs):
        super(EagerMaskedReducalNLL, self).__init__(**kwargs)
        self.params = params

    def call(self, y_true, y_pred):

        raise Exception("关注并联系作者获取该Loss, e-mail:m13541280433@163.com")


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
    #  get feats (batch_size,13,13,3,85)
    #---------------------------------------------------#
    feats = tf.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #---------------------------------------------------#
    #  get box_xy, box_wh
    #---------------------------------------------------#
    box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(grid_shape[..., ::-1], dtype=feats.dtype)
    box_wh = tf.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[..., ::-1], dtype=feats.dtype)

    raise Exception("置信度与分类概率维度的激活方式有变, 关注并联系作者获取, e-mail:m13541280433@163.com")

def box_iou(b1, b2):
    # pred boxes coordinates (left, top) (right, bottom)
    b1 = tf.expand_dims(b1, axis=-2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # true boxes coordinates (left, top) (right, bottom)
    b2 = tf.expand_dims(b2, axis=0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # according to broadcast, pred:(13, 13, 3) → 1, 1 → n
    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

def get_ignore_mask(real_box, pred_box, object_mask, ignore_thresh):
    """
    use to replace tf.while_loop
    """
    ignore_mask = list()
    for i in range(real_box.shape[0]):
        true_box = tf.boolean_mask(real_box[i, ..., :4], object_mask[i])
        iou = box_iou(pred_box[i], true_box)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask.append(best_iou < ignore_thresh)
    return tf.cast(ignore_mask, dtype=true_box.dtype)

#---------------------------------------------------#
#   loss值计算
#---------------------------------------------------#
def yolo_loss(y_true, y_pred, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1):
    num_layers = len(anchors)//3
    #-----------------------------------------------------------#
    #   13x13anchor [142, 110], [192, 243], [459, 401]
    #   26x26anchor [36, 75], [76, 55], [72, 146]
    #   52x52anchor [12, 16], [19, 36], [40, 28]
    #-----------------------------------------------------------#
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    # get input_shape, batch_size
    input_shape = tf.cast(tf.shape(y_pred[0])[1:3] * 32, dtype=y_true[0].dtype)
    batch_size = tf.shape(y_true[0])[0]

    # initialize the total loss
    loss = 0

    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   gain true_boxes masks and one hot labels
        #-----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = smooth_labels(true_class_probs, label_smoothing)

        #-----------------------------------------------------------#
        #   grid        (13,13,1,2)
        #   raw_pred    (m,13,13,3,85)
        #   pred_xy     (m,13,13,3,2)
        #   pred_wh     (m,13,13,3,2)
        #-----------------------------------------------------------#
        pred_xy, pred_wh, box_confidence, box_class_probs = yolo_head(y_pred[l],
                                                                      anchors[anchor_mask[l]],
                                                                      num_classes, input_shape)
        
        #-----------------------------------------------------------#
        #   pred_box    (m,13,13,3,4)
        #-----------------------------------------------------------#
        pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

        #-----------------------------------------------------------#
        #   gain negative samples
        #   ignore_mask (m,13,13,3)
        #-----------------------------------------------------------#
        object_mask_bool = tf.squeeze(tf.cast(object_mask, dtype=tf.bool), axis=-1)
        ignore_mask = tf.TensorArray(dtype=y_true[0].dtype, size=1, dynamic_size=True)

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b])
            # -----------------------------------------------------------#
            #   pred_box    (13,13,3,4)
            #   true_box    (n,4)
            #   iou         (13,13,3,n)
            # -----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            # -----------------------------------------------------------#
            #   best_iou    (13,13,3)
            # -----------------------------------------------------------#
            best_iou = tf.reduce_max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   ignore partial negative samples
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, tf.cast(best_iou < ignore_thresh, dtype=true_box.dtype))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()

        #-----------------------------------------------------------#
        #   calculate positive ciou loss
        #-----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3]*y_true[l][..., 3:4]

        raw_true_box = y_true[l][..., 0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)

        #-----------------------------------------------------------#
        #   calculate total loss
        #-----------------------------------------------------------#
        pos_conf_loss = object_mask[...,-1]*tf.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE,
                                                                         from_logits=True
                                                                         )(object_mask, box_confidence)
        neg_conf_loss = (1-object_mask[...,-1])*tf.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE,
                                                                             from_logits=True
                                                                             )(object_mask, box_confidence)*ignore_mask

        class_loss = EagerMaskedReducalNLL(**{'params': None})(true_class_probs, box_class_probs)

        num_pos = tf.maximum(tf.reduce_sum(tf.cast(object_mask, dtype=tf.float32)), 1)

        location_loss = tf.reduce_sum(ciou_loss)
        pos_conf_loss = tf.reduce_sum(pos_conf_loss)
        neg_conf_loss = tf.reduce_sum(neg_conf_loss)

        loss += (location_loss + pos_conf_loss + neg_conf_loss)/num_pos + class_loss

    loss = loss / num_layers
    return loss
