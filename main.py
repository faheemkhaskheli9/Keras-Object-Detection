import keras
import keras_cv
from tensorflow.keras.utils import Sequence
from keras_cv import visualization, bounding_box
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import json
import os

from dataset import (get_dataset, load_dataset,
                     visualize_dataset, dict_to_tuple)

BATCH_SIZE = 4

BASEPATH = "datasets/train"
IMAGES_PATH = os.path.join(BASEPATH, "images")
LABELS_PATH = os.path.join(BASEPATH, "labels")

classes_labels = ["field","image","text","handwritten","table","checkbox","signature","stamp"]
class_mapping = dict(zip(range(len(classes_labels)), classes_labels))

dataset = get_dataset(IMAGES_PATH, LABELS_PATH)

train_data = dataset.skip(20)
valid_data = dataset.take(20)
test_data = dataset.take(8)

preprocessing = keras.Sequential(
    layers=[
        keras_cv.layers.Resizing(640, 640, interpolation="bilinear",crop_to_aspect_ratio=False,pad_to_aspect_ratio=True, bounding_box_format="xywh")
    ]
)

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomSaturation(0.1),
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        keras_cv.layers.RandomShear(
            x_factor=0.2, y_factor=0.2, bounding_box_format="xywh"
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
        ),
    ]
)

val_ds = valid_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
val_ds = val_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
# test_ds = test_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(BATCH_SIZE)
train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

visualize_dataset(val_ds, bounding_box_format="xywh", value_range=(0, 255), rows=2, cols=2)

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
# train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
# val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

yolov8_preset = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                 "resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2",
                 "mobilenet_v3_small", "mobilenet_v3_large",
                 "csp_darknet_tiny", "csp_darknet_s", "csp_darknet_m", "csp_darknet_l", "csp_darknet_xl",
                 "efficientnetv1_b0", "efficientnetv1_b1", "efficientnetv1_b2", "efficientnetv1_b3", "efficientnetv1_b4", "efficientnetv1_b5", "efficientnetv1_b6", "efficientnetv1_b7",
                 "efficientnetv2_s", "efficientnetv2_m", "efficientnetv2_l", "efficientnetv2_b0", "efficientnetv2_b1", "efficientnetv2_b2", "efficientnetv2_b3",
                 "densenet121", "densenet169", "densenet201",
                 "efficientnetlite_b0", "efficientnetlite_b1", "efficientnetlite_b2", "efficientnetlite_b3", "efficientnetlite_b4",
                 "yolo_v8_xs_backbone", "yolo_v8_s_backbone", "yolo_v8_m_backbone", "yolo_v8_l_backbone", "yolo_v8_xl_backbone",
                 "vitdet_base", "vitdet_large", "vitdet_huge",
                 "resnet50_imagenet", "resnet50_v2_imagenet",
                 "mobilenet_v3_large_imagenet", "mobilenet_v3_small_imagenet",
                 "csp_darknet_tiny_imagenet", "csp_darknet_l_imagenet",
                 "efficientnetv2_s_imagenet", "efficientnetv2_b0_imagenet", "efficientnetv2_b1_imagenet", "efficientnetv2_b2_imagenet",
                 "densenet121_imagenet", "densenet169_imagenet", "densenet201_imagenet",
                 "yolo_v8_xs_backbone_coco", "yolo_v8_s_backbone_coco", "yolo_v8_m_backbone_coco", "yolo_v8_l_backbone_coco", "yolo_v8_xl_backbone_coco",
                 "vitdet_base_sa1b", "vitdet_large_sa1b", "vitdet_huge_sa1b",
                 "yolo_v8_m_pascalvoc"]

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

## Model Config

SPLIT_RATIO = 0.2
LEARNING_RATE = 0.001
EPOCH = 5
GLOBAL_CLIPNORM = 10.0
preset_name = "efficientnetv2_b0_imagenet"

model = keras_cv.models.YOLOV8Detector.from_preset(preset_name,
    num_classes=len(class_mapping),
    bounding_box_format="xywh"
    )

model.compile(optimizer=optimizer,
              classification_loss="binary_crossentropy",
              box_loss="ciou")

checkpoint_filepath = f'{project_name}/tmp/ckpt/checkpoint.model.keras'

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='max',
    save_best_only=True)

backup_restore = keras.callbacks.BackupAndRestore(f"{project_name}/backup", save_freq="epoch", delete_checkpoint=True)

model.fit(train_ds,
          validation_data=val_ds,
          epochs=100,
          callbacks=[
              backup_restore,
              tensorboard_callback,
              model_checkpoint_callback
              ])

