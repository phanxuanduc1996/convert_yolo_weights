import os
import coremltools
import onnxmltools

input_coreml_model = '../../pretrain_models/yolov4/yolov4.mlmodel'
output_onnx_model = '../../pretrain_models/yolov4/yolov4_onnxmltools.onnx'

coreml_model = coremltools.utils.load_spec(input_coreml_model)
onnx_model = onnxmltools.convert_coreml(coreml_model)

onnxmltools.utils.save_model(onnx_model, output_onnx_model)
