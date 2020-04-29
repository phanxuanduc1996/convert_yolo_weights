import tensorflow as tf
import tfcoreml

print("TensorFlow version {}".format(tf.__version__))
print("Eager mode: ", tf.executing_eagerly())
print("Is GPU available: ", tf.test.is_gpu_available())

# tfcoreml.convert(tf_model_path='yolo-obj-416.pb', mlmodel_path='yolo-obj-416.mlmodel',
#                      output_feature_names=['softmax:0'], input_name_shape_dict={'input:0': [1, 416, 416, 3]})

# Input model definition
IMAGE_INPUT_NAME = ['image:0']
IMAGE_INPUT_NAME_SHAPE = {'image:0': [1, 416, 416, 3]}
IMAGE_INPUT_SCALE = 1/255.0
OUTPUT_NAME = ['softmax:0']
MODEL_LABELS = 'labels.txt'
TF_FROZEN_MODEL = '../../pretrain_models/yolov3.pb'
# Output model
CORE_ML_MODEL = '../../pretrain_models/yolov3_tf.mlmodel'

# Convert model and save it as a file
coreml_model = tfcoreml.convert(tf_model_path=TF_FROZEN_MODEL, mlmodel_path=CORE_ML_MODEL, output_feature_names=OUTPUT_NAME, image_input_names=IMAGE_INPUT_NAME, input_name_shape_dict=IMAGE_INPUT_NAME_SHAPE, class_labels=MODEL_LABELS, image_scale=IMAGE_INPUT_SCALE
                                )
