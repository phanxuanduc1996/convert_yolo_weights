from winmltools.utils import save_model
from winmltools import convert_coreml
from coremltools.models.utils import load_spec


input_coreml_model = '../webservice/pretrain_models/water_meter_yolov2_tiny/yolo-obj-416-water_yolov2_tiny_15000.mlmodel'
output_onnx_model = '../webservice/pretrain_models/water_meter_yolov2_tiny/yolo-obj-416-water_yolov2_tiny_15000_winmltools.onnx'

model_coreml = load_spec(input_coreml_model)
model_onnx = convert_coreml(
    model_coreml, 8, name='yolo-obj-416-water_yolov2_tiny_15000_winmltools')

save_model(model_onnx, output_onnx_model)
