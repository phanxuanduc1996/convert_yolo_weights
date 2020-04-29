from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
import os


# Load the ONNX model
onnx_model = onnx.load(
    '../pretrain_models/yolov3.onnx')
print(onnx_model)
