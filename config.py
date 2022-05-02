# -*- coding: UTF-8 -*-
'''
@Project ：Yolo-lite-based-on-ShuffleNet
@File    ：config.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
from _utils.utils import get_classes, get_anchors


# ===annotation===
xml_root = 'coco数据集标注文件本地目录'
image_root = 'coco数据集本地目录'


# ===generator===
annotation_path = 'get_annotation.py获取的txt文件路径(置于model_data中)'
classes_path = '定义检测目标的类别的txt文件路径(置于model_data中)'
anchors_path = 'kmeans_for_anchors.py获取的先验框尺寸的txt文件路径(置于model_data中)'
train_split = 0.7
input_size = (416, 416)

# ===model===
backbone = 'shufflenet'
input_shape = (None, None, 3)  # support unknown input shape
anchors = get_anchors(anchors_path)
class_names = get_classes(classes_path)

# ===training===
Epoches = 150
batch_size = 8
learning_rate = 1e-4
warmup_learning_rate = 1e-5
min_learning_rate = 1e-7
ckpt_path = '.\\tf_models\\checkpoint'
cosine_scheduler = True

# ===prediction===
iou = 0.35
score = 0.35
max_boxes = 100
font_color = (0, 255, 0)
rect_color = (0, 0, 255)
thickness = 0.5
font_path = '.\\font\\simhei.ttf'
sample_path = ".\\sample\\Batch{}.jpg"

defaults = {
    "model_path": '.\\tf_models\\checkpoint',
    "anchors_path": 'kmeans_for_anchors.py获取的先验框尺寸的txt文件路径(置于model_data中)',
    "classes_path": '定义检测目标的类别的txt文件路径(置于model_data中)',
    "backbone": 'shufflenet',
    "alpha": 1,
    "score": 0.35,
    "iou": 0.35,
    "max_boxes": 100,
    "model_image_size": (416, 416),
    "letterbox_image": False
}
