# Convert .weights to .pt
	python3  -c "from models import *; convert('cfg/yolov4.cfg', 'weights/yolov4.pt')"

# Detect with .pt model.
	python3 detect.py --cfg cfg/yolov4.cfg --weights yolov4.pt
