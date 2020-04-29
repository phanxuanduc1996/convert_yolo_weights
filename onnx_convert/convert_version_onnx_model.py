from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import onnx
from onnx import version_converter, helper


model_path = '../../pretrain_models/yolov3.onnx'
original_model = onnx.load(model_path)

converted_model = version_converter.convert_version(original_model, 8)
onnx.save(converted_model,
          '../../pretrain_models/yolov3_version_8.onnx')
