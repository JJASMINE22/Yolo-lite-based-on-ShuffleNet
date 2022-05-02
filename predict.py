# -*- coding: UTF-8 -*-
'''
@Project ：Yolo-lite-based-on-ShuffleNet
@File    ：predict.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from nets.yolo4 import yolo_body, yolo_eval
from _utils.utils import letterbox_image
import config as cfg


class YOLO:
    def __init__(self):
        self.__dict__.update(cfg.defaults)
        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        self.model = yolo_body((None, None, 3), num_anchors // 3, num_classes, self.backbone)
        ckpt = tf.train.Checkpoint(bridge=self.model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, self.model_path, max_to_keep=5)

        # if the checkpoint exists, restore the latest checkpoint and load the model
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored!!')

    def get_class(self):
        with open(self.classes_path, 'r') as f:
            class_names = f.readlines()
        class_names = [cls.strip() for cls in class_names]
        return class_names

    def get_anchors(self):
        with open(self.anchors_path, 'r') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def detect_image(self, image):
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1], self.model_image_size[0]), Image.BICUBIC)

        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        outputs = self.model(image_data)
        out_boxes, out_scores, out_classes = yolo_eval(yolo_outputs=outputs,
                                                       anchors=self.anchors,
                                                       num_classes=len(self.class_names),
                                                       image_shape=[image.size[1], image.size[0]],
                                                       max_boxes=self.max_boxes,
                                                       score_threshold=self.score,
                                                       iou_threshold=self.iou)

        return out_classes, out_boxes, out_scores

if __name__ == '__main__':

    yolo = YOLO()
    while True:
        file_path = input('enter image path:')
        try:
            image = Image.open(file_path)
        except FileNotFoundError:
            print('the path does not exist')
            continue
        else:
            out_classes, out_boxes, out_scores = yolo.detect_image(image)

            out_boxes = out_boxes[0].numpy()
            out_scores = out_scores[0].numpy()
            out_classes = out_classes[0].numpy()

            for i, coordinate in enumerate(out_boxes.astype('int')):
                left, top = list(reversed(coordinate[:2]))
                right, bottom = list(reversed(coordinate[2:]))

                font = ImageFont.truetype(font=cfg.font_path,
                                          size=np.floor(4e-2 * image.size[1] + 0.5).astype('int32'))

                label = '{:s}: {:.2f}'.format(cfg.class_names[out_classes[i]],
                                              out_scores[i])

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
            image.show()
