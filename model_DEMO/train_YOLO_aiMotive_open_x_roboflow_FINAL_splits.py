# Imports
import torch
from ultralytics import YOLO
import cv2
import os
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from PIL import Image
from matplotlib import pyplot as plt


# Path to the custom YAML configuration (after you modify it to only include speed signs)
yaml_config = 'datasets/FINAL_val_splits/config.yaml'
model = YOLO('yolov8s.pt')  # Replace 'yolov8s.pt' with your desired YOLOv8 version

model.train(
    data=yaml_config,
    epochs=150,
    imgsz=1024, # 640
    batch=8, # 8
    name='yolo_speedSigns_aiMotive_open_FINAL_splits_v2',
    cache=True,
    patience=10,
    fliplr=0.0,
    flipud=0.0,
    degrees=0.1,
    mixup = 0.1,
    mosaic = 0.4,
    shear = 0.1,
    scale=0.1
)
