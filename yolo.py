# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
import config as cfg
from PIL import Image, ImageDraw, ImageFont
from nets.yolo4 import yolo_body, yolo_eval
from nets.loss import yolo_loss

class YOLO:
    def __init__(self,
                 input_shape: tuple,
                 anchors: np.ndarray,
                 classes_names: list,
                 learning_rate: float,
                 score_thresh: float,
                 iou_thresh: float,
                 max_boxes: int,
                 backbone: str):

        self.anchors = anchors
        self.max_boxes = max_boxes
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.classes_names = classes_names
        self.num_anchors = anchors.__len__()
        self.num_classes = classes_names.__len__()
        self.learning_rate = learning_rate

        # model structure
        self.model = yolo_body(input_shape, self.num_anchors//3, self.num_classes, backbone)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # record train&validate loss
        self.train_loss = tf.keras.metrics.Mean()
        self.val_loss = tf.keras.metrics.Mean()
        # record the confidences of 13x13/26x26/52x52 fields
        self.train_conf_minimum = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_conf_medium = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_conf_maximum = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_conf_minimum = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_conf_medium = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_conf_maximum = tf.keras.metrics.SparseCategoricalAccuracy()
        # record the classification probs of 13x13/26x26/52x52 fields
        self.train_class_minimum = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_class_medium = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_class_maximum = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_class_minimum = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_class_medium = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_class_maximum = tf.keras.metrics.SparseCategoricalAccuracy()

    @tf.function
    # eager execution
    def train(self, sources, targets):
        with tf.GradientTape() as tape:
            logits = self.model(sources)
            loss = yolo_loss(targets, logits, self.anchors, self.num_classes)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        logits = [tf.reshape(logit, shape=[tf.shape(logit)[0], tf.shape(logit)[1], tf.shape(logit)[2],
                                           self.num_anchors//3, -1]) for logit in logits]

        prob_confs = [tf.sigmoid(logit[..., 4:5]) for logit in logits]
        real_confs = [target[..., 4:5] for target in targets]
        self.train_conf_minimum(real_confs[0], prob_confs[0])
        self.train_conf_medium(real_confs[1], prob_confs[1])
        self.train_conf_maximum(real_confs[2], prob_confs[2])
        object_masks = [tf.squeeze(tf.cast(real_conf, dtype=tf.bool), axis=-1)
                        for real_conf in real_confs]
        prob_classes = [tf.boolean_mask(logit[..., 5:], mask)
                        for logit, mask in zip(logits, object_masks)]
        real_classes = [tf.boolean_mask(target[..., 5:], mask)
                        for target, mask in zip(targets, object_masks)]
        self.train_class_minimum(tf.argmax(real_classes[0], axis=-1), prob_classes[0])
        self.train_class_medium(tf.argmax(real_classes[1], axis=-1), prob_classes[1])
        self.train_class_maximum(tf.argmax(real_classes[2], axis=-1), prob_classes[2])

    @tf.function
    # eager execution
    def validate(self, sources, targets):

        logits = self.model(sources)
        loss = yolo_loss(targets, logits, self.anchors, self.num_classes)

        self.val_loss(loss)
        logits = [tf.reshape(logit, shape=[tf.shape(logit)[0], tf.shape(logit)[1], tf.shape(logit)[2],
                                           self.num_anchors//3, -1]) for logit in logits]

        prob_confs = [tf.sigmoid(logit[..., 4:5]) for logit in logits]
        real_confs = [target[..., 4:5] for target in targets]
        self.val_conf_minimum(real_confs[0], prob_confs[0])
        self.val_conf_medium(real_confs[1], prob_confs[1])
        self.val_conf_maximum(real_confs[2], prob_confs[2])
        object_masks = [tf.squeeze(tf.cast(real_conf, dtype=tf.bool), axis=-1)
                        for real_conf in real_confs]
        prob_classes = [tf.boolean_mask(logit[..., 5:], mask)
                        for logit, mask in zip(logits, object_masks)]
        real_classes = [tf.boolean_mask(target[..., 5:], mask)
                        for target, mask in zip(targets, object_masks)]
        self.val_class_minimum(tf.argmax(real_classes[0], axis=-1), prob_classes[0])
        self.val_class_medium(tf.argmax(real_classes[1], axis=-1), prob_classes[1])
        self.val_class_maximum(tf.argmax(real_classes[2], axis=-1), prob_classes[2])

    def generate_sample(self, sources, batch):
        """
        Drawing and labeling
        """
        logits = self.model(sources)
        image_size = tf.shape(sources)[1:3]
        out_boxes, out_scores, out_classes = yolo_eval(yolo_outputs=logits,
                                                       anchors=self.anchors,
                                                       num_classes=self.num_classes,
                                                       image_shape=image_size,
                                                       max_boxes=self.max_boxes,
                                                       score_threshold=self.score_thresh,
                                                       iou_threshold=self.iou_thresh)

        out_boxes = [out_box.numpy() for out_box in out_boxes]
        out_scores = [out_score.numpy() for out_score in out_scores]
        out_classes = [out_class.numpy() for out_class in out_classes]

        index = np.random.choice(np.shape(sources)[0], 1)[0]
        source = sources[index]
        image = Image.fromarray(np.uint8(source * 255))

        for i, coordinate in enumerate(out_boxes[index].astype('int')):
            left, top = list(reversed(coordinate[:2]))
            right, bottom = list(reversed(coordinate[2:]))

            font = ImageFont.truetype(font=cfg.font_path,
                                      size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

            label = '{:s}: {:.2f}'.format(self.classes_names[out_classes[index][i]],
                                          out_scores[index][i])

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle(list(reversed(coordinate[:2])) + list(reversed(coordinate[2:])),
                           outline=cfg.rect_color, width=int(2 * cfg.thickness))

            draw.text(text_origin, str(label, 'UTF-8'),
                      fill=cfg.font_color, font=font)
            del draw
        image.save(cfg.sample_path.format(batch), quality=95, subsampling=0)
