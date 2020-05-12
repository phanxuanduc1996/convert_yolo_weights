# tensorflow version 1.13.1

from tensorflow.contrib import lite
converter = lite.TFLiteConverter.from_keras_model_file(
    '../../pretrain_models/yolov3/yolov3.h5', input_shapes={"image_input": [1, 608, 608, 3]})

model = converter.convert()
open("../../pretrain_models/yolov3/yolov3_keras.tflite", "wb").write(model)
