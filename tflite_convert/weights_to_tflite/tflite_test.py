import time
import cv2
import numpy as np
import argparse
from predict import preprocess_image
from predict import handle_predictions
from predict import draw_boxes
from yolo_head import yolo_head
from coco_labels import COCOLabels
import tensorflow as tf
import os
import sys
sys.path.append(os.path.abspath('./src'))


parser = argparse.ArgumentParser(description='Keras to TF-Lite converter')
parser.add_argument('--image_input', default='input_image.jpg',
                    help='Path to a image test file.')
parser.add_argument('--image_output', default='output_image.jpg',
                    help='Path to image predict file to save.')
args = parser.parse_args()

data_dir = '../../pretrain_models/'
model_path = os.path.join(
    os.getcwd(), '../../pretrain_models/yolov3.tflite')
interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

config = {}
config['width'] = width
config['height'] = height
config['labels'] = COCOLabels.all(data_dir)
config['colors'] = COCOLabels.colors(data_dir)
start = time.time()
frame = cv2.imread(args.image_input)
image, image_data = preprocess_image(frame, (height, width))

interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
output = [interpreter.get_tensor(output_details[i]['index'])
          for i in range(len(output_details))]
predictions = yolo_head(output, num_classes=10, input_dims=(width, height))

boxes, classes, scores = handle_predictions(predictions,
                                            confidence=0.3,
                                            iou_threshold=0.5)
predict, conf = draw_boxes(image, boxes, classes, scores, config)
print(predict, conf)
image = np.array(image)

end = time.time()
print("Inference time: {:.2f}s".format(end - start))

# Display the resulting frame
cv2.imwrite(args.image_output, image)
