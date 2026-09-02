import sys

import fastapi
import ultralytics
from ultralytics import YOLO

print("Python:", sys.version)
print("FastAPI:", fastapi.__version__)
print("Ultralytics:", ultralytics.__version__)

YOLO("yolov8n.pt")
print("Model weights ready: yolov8n.pt")
