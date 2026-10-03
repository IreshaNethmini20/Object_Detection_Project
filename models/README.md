# Pretrained model files

This project uses YOLOv4 pretrained on COCO. It does not train YOLO from
scratch or create a custom dataset. The existing `yolov4.cfg` and 80 COCO
class labels in `coco.names` are included unchanged.

Download **yolov4.weights** from the official
[AlexeyAB Darknet YOLOv4 release](https://github.com/AlexeyAB/darknet/releases/tag/yolov4).
Choose the full YOLOv4 weights, rather than tiny or another model variant.
Place the downloaded file at `models/yolov4.weights`.

Weights are large and excluded from Git. No download happens automatically.
Default model paths are relative to the project code, including on Windows.
Restart FastAPI after adding or replacing model files.
