#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert YOLO keras model to ONNX model
"""

import os
import sys
import argparse
from keras import backend as K
from keras.models import load_model
import keras2onnx
import onnx
from yolo4_utils import get_custom_objects

os.environ['TF_KERAS'] = '1'


def onnx_convert(keras_model_file, output_file):
    custom_object_dict = get_custom_objects()
    model = load_model(keras_model_file, custom_objects=custom_object_dict)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # save converted onnx model
    onnx.save_model(onnx_model, output_file)


def main():
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, description='Convert YOLO tf.keras model to ONNX model')
    parser.add_argument('--keras_model_file', type=str,
                        help='path to keras model file', default='../../../pretrain_models/yolov4/yolov4.h5')
    parser.add_argument('--output_file', type=str,
                        help='output onnx model file', default='../../../pretrain_models/yolov4/yolov4.onnx')

    args = parser.parse_args()

    onnx_convert(args.keras_model_file, args.output_file)


if __name__ == '__main__':
    main()
