import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    '../../pretrain_models/yolov3.tflite')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open('../../pretrain_models/yolov3_quantized.tflite',
     "wb").write(tflite_quantized_model)
