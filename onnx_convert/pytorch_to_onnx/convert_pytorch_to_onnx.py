import torch.onnx
import torch
from torch.autograd import Variable
from models import Darknet

# # Load the trained model from file
# trained_model = Darknet(
#     '../webservice/pretrain_models/water_meter/yolo-obj-416-water.cfg')
# trained_model.load_state_dict(torch.load(
#     '../webservice/pretrain_models/water_meter/yolo-obj-416-water_15000.pt'))


device = torch.device('cpu')
model = torch.load(
    '../../pretrain_models/yolov3.pt', map_location=device)

dummy_input = torch.randn(1, 3, 416, 416)
torch.onnx.export(model, dummy_input,
                  '../../pretrain_models/yolov3_pt.onnx')
