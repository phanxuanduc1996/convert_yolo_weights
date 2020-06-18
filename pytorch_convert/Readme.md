# Convert .weights to .pt
	python3  -c "from models import *; convert('../pretrain_models/yolov4/yolov4.cfg', '../pretrain_models/yolov4/yolov4.weights')"

# Detect with .pt model.
	python3 detect.py --cfg ../pretrain_models/yolov4/yolov4.cfg --weights ../pretrain_models/yolov4/yolov4.pt
