# -*- coding: UTF-8 -*-
'''
@Project ：Yolo-lite-based-on-ShuffleNet
@File    ：train.py
@IDE     ：PyCharm
@Author  ：XinYi Huang
'''
import os
import numpy as np
import tensorflow as tf
import config as cfg
from yolo import YOLO
from _utils.generate import Generator
from _utils.utils import WarmUpCosineDecayScheduler
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.InteractiveSession(config=config)

if __name__ == '__main__':

    Yolo = YOLO(input_shape=cfg.input_shape,
                anchors=cfg.anchors,
                classes_names=cfg.class_names,
                learning_rate=cfg.learning_rate,
                max_boxes=cfg.max_boxes,
                backbone=cfg.backbone,
                score_thresh=cfg.score,
                iou_thresh=cfg.iou)

    data_gen = Generator(annotation_path=cfg.annotation_path,
                         input_size=cfg.input_size,
                         batch_size=cfg.batch_size,
                         train_split=cfg.train_split,
                         anchors=cfg.anchors,
                         num_classes=cfg.class_names.__len__())

    train_gen = data_gen.generate(training=True)
    validate_gen = data_gen.generate(training=False)

    if not os.path.exists(cfg.ckpt_path):
        os.makedirs(cfg.ckpt_path)

    ckpt = tf.train.Checkpoint(bridge=Yolo.model,
                               optimizer=Yolo.optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_path, max_to_keep=5)

    # if the checkpoint exists, restore the latest checkpoint and load the model
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    if cfg.cosine_scheduler:
        total_steps = data_gen.get_train_len() * cfg.Epoches
        warmup_steps = int(data_gen.get_train_len() * cfg.Epoches * 0.2)
        hold_steps = data_gen.get_train_len() * data_gen.batch_size
        reduce_lr = WarmUpCosineDecayScheduler(global_interval_steps=total_steps,
                                               warmup_interval_steps=warmup_steps,
                                               hold_interval_steps=hold_steps,
                                               learning_rate_base=cfg.learning_rate,
                                               warmup_learning_rate=cfg.warmup_learning_rate,
                                               min_learning_rate=cfg.min_learning_rate,
                                               verbose=0)
    for epoch in range(cfg.Epoches):
        # ----training----
        print('------start training------')
        for i in range(data_gen.get_train_len()):
            sources, targets = next(train_gen)
            if cfg.cosine_scheduler:
                learning_rate = reduce_lr.batch_begin()
                Yolo.optimizer.learning_rate = learning_rate
            Yolo.train(sources, targets)
            if not (i+1) % 50:
                Yolo.generate_sample(sources, i+1)
                print('yolo_loss: {}\n'.format(Yolo.train_loss.result().numpy()),
                      'conf_acc: {}\n'.format(np.mean([Yolo.train_conf_minimum.result().numpy()*100,
                                                       Yolo.train_conf_medium.result().numpy()*100,
                                                       Yolo.train_conf_maximum.result().numpy()*100])),
                      'class_acc: {}\n'.format(np.mean([Yolo.train_class_minimum.result().numpy()*100,
                                                        Yolo.train_class_medium.result().numpy()*100,
                                                        Yolo.train_class_maximum.result().numpy()*100]))
                      )
        # ----validating----
        print('------start validating------')
        for i in range(data_gen.get_val_len()):
            sources, targets = next(validate_gen)
            Yolo.validate(sources, targets)
            if not (i+1) % 50:
                print('yolo_loss: {}\n'.format(Yolo.val_loss.result().numpy()),
                      'conf_acc: {}\n'.format(np.mean([Yolo.val_conf_minimum.result().numpy()*100,
                                                       Yolo.val_conf_medium.result().numpy()*100,
                                                       Yolo.val_conf_maximum.result().numpy()*100])),
                      'class_acc: {}\n'.format(np.mean([Yolo.val_class_minimum.result().numpy()*100,
                                                        Yolo.val_class_medium.result().numpy()*100,
                                                        Yolo.val_class_maximum.result().numpy()*100]))
                      )

        print(f'Epoch {epoch + 1}\n',
              f'train_yolo_loss: {Yolo.train_loss.result().numpy()}\n',
              'train_conf_acc: {}\n'.format(np.mean([Yolo.train_conf_minimum.result().numpy()*100,
                                                     Yolo.train_conf_medium.result().numpy()*100,
                                                     Yolo.train_conf_maximum.result().numpy()*100])),
              'train_class_acc: {}\n'.format(np.mean([Yolo.train_class_minimum.result().numpy()*100,
                                                      Yolo.train_class_medium.result().numpy()*100,
                                                      Yolo.train_class_maximum.result().numpy()*100])),
              f'val_yolo_loss: {Yolo.val_loss.result().numpy()}\n',
              'val_conf_acc: {}\n'.format(np.mean([Yolo.val_conf_minimum.result().numpy()*100,
                                                   Yolo.val_conf_medium.result().numpy()*100,
                                                   Yolo.val_conf_maximum.result().numpy()*100])),
              'val_class_acc: {}\n'.format(np.mean([Yolo.val_class_minimum.result().numpy()*100,
                                                    Yolo.val_class_medium.result().numpy()*100,
                                                    Yolo.val_class_maximum.result().numpy()*100]))
              )
        ckpt_save_path = ckpt_manager.save()

        Yolo.train_loss.reset_states()
        Yolo.val_loss.reset_states()
        Yolo.train_conf_maximum.reset_states()
        Yolo.train_conf_medium.reset_states()
        Yolo.train_conf_minimum.reset_states()
        Yolo.train_class_maximum.reset_states()
        Yolo.train_class_medium.reset_states()
        Yolo.train_class_minimum.reset_states()
        Yolo.val_class_maximum.reset_states()
        Yolo.val_class_medium.reset_states()
        Yolo.val_class_minimum.reset_states()
        Yolo.val_class_maximum.reset_states()
        Yolo.val_class_medium.reset_states()
        Yolo.val_class_minimum.reset_states()