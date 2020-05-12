import sys
import os
sys.path.append(os.path.abspath('./src'))
from darknet import darknet_base
import argparse
from tensorflow.keras import Input, Model
import tensorflow as tf


parser = argparse.ArgumentParser(description='Keras to TF-Lite converter')
parser.add_argument('--h5_path', default='../../pretrain_models/yolov3/yolov3.h5',
                    help='Path to Darknet h5 weights file.')
parser.add_argument('--output_path', default='../../pretrain_models/yolov3/yolov3_weights.tflite',
                    help='Path to output Keras model file.')
args = parser.parse_args()


inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(
    inputs, yolo_version='yolov3', data_dir='../../pretrain_models/yolov3', include_yolo_head=False)

model = Model(inputs, outputs)
model_path = args.h5_path

tf.keras.models.save_model(model, model_path, overwrite=True)
model.summary()
# Sanity check to see if model loads properly
# NOTE: See https://github.com/keras-team/keras/issues/4609#issuecomment-329292173
# on why we have to pass in `tf: tf` in `custom_objects`
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'tf': tf})

converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model_path,
                                                                  input_shapes={'input_1': [1, config['width'], config['height'], 3]})
tflite_model = converter.convert()
open(args.output_path, "wb").write(tflite_model)
