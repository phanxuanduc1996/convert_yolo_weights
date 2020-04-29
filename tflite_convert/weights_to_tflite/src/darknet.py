import os

import numpy as np
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    Lambda, LeakyReLU, GlobalMaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import MaxPool2D, MaxPooling2D

from coco_labels import COCOLabels
from utils.parser import Parser
from yolo_layer import YOLOLayer

# TODO: It seems like turning on eager execution always gives different values during
# TODO: inference. Weirdly, it also loads the network very fast compared to non-eager.
# TODO: It could be that in eager mode, the weights are not loaded. Need to verify
# TODO: this.
# tf.enable_eager_execution()


def darknet_base(inputs, YOLO_VERSION='yolov3', data_dir='./weights', include_yolo_head=True):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param inputs: Input tensor
    :param include_yolo_head: Includes the YOLO head
    :return: A list of output layers and the network config
    """
    weights_file = open(os.path.join(data_dir, '{}.weights'.format(YOLO_VERSION)), 'rb')
    major, minor, revision = np.ndarray(
        shape=(3,), dtype='int32', buffer=weights_file.read(12))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    path = os.path.join(data_dir, '{}.cfg'.format(YOLO_VERSION))
    blocks = Parser.parse_cfg(path)
    x, layers, yolo_layers = inputs, [], []
    ptr = 0
    config = {}

    for block in blocks:
        block_type = block['type']

        if block_type == 'net':
            config = _read_net_config(block, data_dir)

        elif block_type == 'convolutional':
            x, layers, yolo_layers, ptr = _build_conv_layer(x, block, layers, yolo_layers, ptr, config, weights_file)

        elif block_type == 'shortcut':
            x, layers, yolo_layers, ptr = _build_shortcut_layer(x, block, layers, yolo_layers, ptr)

        elif block_type == 'yolo':
            x, layers, yolo_layers, ptr = _build_yolo_layer(x, block, layers, yolo_layers, ptr, config)

        elif block_type == 'route':
            x, layers, yolo_layers, ptr = _build_route_layer(x, block, layers, yolo_layers, ptr)

        elif block_type == 'upsample':
            x, layers, yolo_layers, ptr = _build_upsample_layer(x, block, layers, yolo_layers, ptr)

        elif block_type == 'maxpool':
            x, layers, yolo_layers, ptr = _build_maxpool_layer(x, block, layers, yolo_layers, ptr)

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    _verify_weights_completed_consumed(ptr, weights_file)

    if include_yolo_head:
        output_layers = yolo_layers
        return tf.keras.layers.Concatenate(axis=1)(output_layers), config
    else:
        output_layers = [layers[i - 1] for i in range(len(layers)) if layers[i] is None]

        # NOTE: Apparently TFLite doesn't like Concatenate.
        # return tf.keras.layers.Concatenate(axis=1)(output_layers), config
        return output_layers, config


def _read_net_config(block, data_dir):
    width = int(block['width'])
    height = int(block['height'])
    channels = int(block['channels'])
    decay = float(block['decay'])

    labels = COCOLabels.all(data_dir)
    colors = COCOLabels.colors(data_dir)

    return {
        'width': width,
        'height': height,
        'channels': channels,
        'labels': labels,
        'colors': colors,
        'decay': decay
    }


def _build_conv_layer(x, block, layers, outputs, ptr, config, weights_file):
    stride = int(block['stride'])
    filters = int(block['filters'])
    kernel_size = int(block['size'])
    pad = int(block['pad'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    use_batch_normalization = 'batch_normalize' in block

    # Darknet serializes convolutional weights as:
    # [bias/beta, [gamma, mean, variance], conv_weights]
    prev_layer_shape = K.int_shape(x)
    weights_shape = (kernel_size, kernel_size, prev_layer_shape[-1], filters)
    darknet_w_shape = (filters, weights_shape[2], kernel_size, kernel_size)
    weights_size = np.product(weights_shape)

    # number of filters * 4 bytes
    conv_bias = np.ndarray(
        shape=(filters,),
        dtype='float32',
        buffer=weights_file.read(filters * 4)
    )
    ptr += filters

    bn_weights_list = []
    if use_batch_normalization:
        # [gamma, mean, variance] * filters * 4 bytes
        bn_weights = np.ndarray(
            shape=(3, filters),
            dtype='float32',
            buffer=weights_file.read(3 * filters * 4)
        )
        ptr += 3 * filters

        bn_weights_list = [
            bn_weights[0],  # scale gamma
            conv_bias,  # shift beta
            bn_weights[1],  # running mean
            bn_weights[2]  # running var
        ]

    conv_weights = np.ndarray(
        shape=darknet_w_shape,
        dtype='float32',
        buffer=weights_file.read(weights_size * 4))
    ptr += weights_size

    # DarkNet conv_weights are serialized Caffe-style:
    # (out_dim, in_dim, height, width)
    # We would like to set these to Tensorflow order:
    # (height, width, in_dim, out_dim)
    conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
    if use_batch_normalization:
        conv_weights = [conv_weights]
    else:
        conv_weights = [conv_weights, conv_bias]

    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               use_bias=not use_batch_normalization,
               activation='linear',
               kernel_regularizer=l2(config['decay']),
               weights=conv_weights)(x)

    if use_batch_normalization:
        x = BatchNormalization(weights=bn_weights_list)(x)

    assert block['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(block['activation'])

    if block['activation'] == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers, outputs, ptr


def _build_upsample_layer(x, block, layers, outputs, ptr):
    stride = int(block['stride'])

    # NOTE: Alternative way of defining Upsample2D
    x = Lambda(lambda _x: tf.image.resize_bilinear(_x, (stride * tf.shape(_x)[1], stride * tf.shape(_x)[2])))(x)
    # x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers, outputs, ptr


def _build_maxpool_layer(x, block, layers, outputs, ptr):
    stride = int(block['stride'])
    size = int(block['size'])

    x = MaxPooling2D(pool_size=(size, size),
                     strides=(stride, stride),
                     padding='same')(x)
    layers.append(x)

    return x, layers, outputs, ptr


def _build_route_layer(_x, block, layers, outputs, ptr):
    selected_layers = [layers[int(l)] for l in block['layers'].split(',')]

    if len(selected_layers) == 1:
        x = selected_layers[0]
        layers.append(x)

        return x, layers, outputs, ptr

    elif len(selected_layers) == 2:
        x = Concatenate(axis=3)(selected_layers)
        layers.append(x)

        return x, layers, outputs, ptr

    else:
        raise ValueError('Invalid number of layers: {}'.format(len(selected_layers)))


def _build_shortcut_layer(x, block, layers, outputs, ptr):
    from_layer = layers[int(block['from'])]
    x = Add()([from_layer, x])

    assert block['activation'] == 'linear', 'Invalid activation: {}'.format(block['activation'])
    layers.append(x)

    return x, layers, outputs, ptr


def _build_yolo_layer(x, block, layers, outputs, ptr, config):
    # Read indices of masks
    masks = [int(m) for m in block['mask'].split(',')]
    # Anchors used based on mask indices
    anchors = [a for a in block['anchors'].split(',  ')]
    anchors = [anchors[i] for i in range(len(anchors)) if i in masks]
    anchors = [[int(a) for a in anchor.split(',')] for anchor in anchors]
    classes = int(block['classes'])

    x = YOLOLayer(num_classes=classes, anchors=anchors, input_dims=(config['width'], config['height']))(x)
    outputs.append(x)
    # NOTE: Here we append None to specify that the preceding layer is a output layer
    layers.append(None)

    return x, layers, outputs, ptr


def _verify_weights_completed_consumed(ptr, weights_file):
    remaining_weights = len(weights_file.read()) // 4
    weights_file.close()
    percentage = int((ptr / (ptr + remaining_weights)) * 100)
    print('Read {}% from Darknet weights.'.format(percentage))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))
    else:
        print('Weights loaded successfully!')
