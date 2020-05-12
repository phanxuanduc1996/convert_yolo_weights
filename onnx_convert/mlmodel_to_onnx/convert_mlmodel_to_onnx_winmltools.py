from winmltools.utils import save_model
from winmltools import convert_coreml
from coremltools.models.utils import load_spec


input_coreml_model = '../../pretrain_models/yolov4/yolov4.mlmodel'
output_onnx_model = '../../pretrain_models/yolov4/yolov4_winmltools.onnx'

model_coreml = load_spec(input_coreml_model)
model_onnx = convert_coreml(
    model_coreml, 8, name='yolov4_winmltools')

save_model(model_onnx, output_onnx_model)
