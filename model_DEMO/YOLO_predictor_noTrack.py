# Imports
from symtable import Class

import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import os
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import re
from imageio import mimsave
import json
from enum import Enum
from copy import deepcopy
from network import ClassifierNet
import numpy as np
import torch.nn.functional as F
import imageio


os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

class TrafficSignEUSpeedLimit(Enum):
    EU_SPEEDLIMIT_5 = 0
    EU_SPEEDLIMIT_10 = 1
    EU_SPEEDLIMIT_20 = 2
    EU_SPEEDLIMIT_30 = 3
    EU_SPEEDLIMIT_40 = 4
    EU_SPEEDLIMIT_50 = 5
    EU_SPEEDLIMIT_60 = 6
    EU_SPEEDLIMIT_70 = 7
    EU_SPEEDLIMIT_80 = 8
    EU_SPEEDLIMIT_90 = 9
    EU_SPEEDLIMIT_100 = 10
    EU_SPEEDLIMIT_110 = 11
    EU_SPEEDLIMIT_120 = 12
    EU_SPEEDLIMIT_130 = 13

def sections_from_dir(img_dir):
    grouped_images = {}
    regex_pattern = r'(\d{8}-\d{6}-\d{2}\.\d{2}\.\d{2}-\d{2}\.\d{2}\.\d{2}@[A-Za-z]+)_.*_F_MIDRANGECAM.*'
    for file_name in os.listdir(img_dir):
        if file_name.endswith(('.jpg', '.png')):
            match = re.match(regex_pattern, file_name)
            if match:
                section_id = match.group(1)

                if section_id not in grouped_images:
                    grouped_images[section_id] = []
                grouped_images[section_id].append(os.path.join(img_dir, file_name))
    # Ordering the lists:
    for section_id in grouped_images:
        # Use natsort to sort the files in a natural order (taking into account numbers in filenames)
        grouped_images[section_id] = sorted(grouped_images[section_id])

    return grouped_images

def create_gif_and_display(yolo_results, crop_results, output_dir, event=False):
    frames = []
    for i, r in enumerate(yolo_results):
        img = cv2.cvtColor(r.orig_img, cv2.COLOR_BGR2RGB)
        annotator = Annotator(img)

        boxes = r.boxes

        # Annotate the image with detected boxes and confidence scores
        for j, box in enumerate(boxes):
            confidence = box.conf[0]
            b = box.xyxy[0]  # Get box coordinates in (left, top, right, bottom) format
            c = box.cls

            crop_conf = F.softmax(crop_results[i][j], dim=1)
            annotator.box_label(b, f'Y: {model.names[int(c)][-2:]} {confidence.item()}'
                                   f'CNN: {model.names[np.argmax(crop_results[i][j]).item()][-2:]} {torch.max(crop_conf).item()}')

        img = annotator.result()  # Get the annotated image result

        # Append the processed frame to the list
        frames.append(img)
        i += 1

    # Create and display the GIF from the processed frames
    if frames:
        # Saving as gif
        gif_path = os.path.join(output_dir, 'predictions_gifs')
        os.makedirs(gif_path, exist_ok=True)
        mimsave(os.path.join(gif_path, f"{section}_output_{'eventBased' if event else ''}.gif"), frames, duration=1)

        # Saving as plots
        jpg_output_dir = os.path.join(output_dir, 'predictions_jpgs', f"{section}_output_{'eventBased' if event else ''}")
        os.makedirs(jpg_output_dir, exist_ok=True)

        # Save each frame as a separate JPG
        for i, frame in enumerate(frames):
            frame_path = os.path.join(jpg_output_dir, f"frame_{i:03d}.jpg")
            imageio.imwrite(frame_path, frame)
    else:
        print(f"No detections for section: {section}")

class CustomBox:
    def __init__(self, cls=None, conf=None, xyxy=None):
        self.cls = cls if cls is not None else torch.tensor([])
        self.conf = conf if conf is not None else torch.tensor([])
        self.xyxy = xyxy if xyxy is not None else torch.tensor([])

    def __iter__(self):
        return iter([])


if __name__ == '__main__':
    create_event_based = False
    crop_model_path = 'best_model_4_FINAL_splits.pth'
    model_path = 'i/deleteme_regurarly/runs/detect/yolo_speedSigns_aiMotive_open_FINAL_v12/weights/best.pt'
    output_dir = '/traffic_sign_classificator_2_stage/YOLO_predictions/'
    working_dir = '/deleteme_regurarly/2_stage_speed_limit_classifier/aiMotive_open_archive/aiMotive_open_FINAL_validation_splits_without_insider_data/'

    print(f'Loading model from {model_path}')
    model = YOLO(model_path)
    pred_input_dir = os.path.join(working_dir, 'val/images/')
    gt_input_dir = os.path.join(working_dir, 'val/labels/')

    classes = TrafficSignEUSpeedLimit
    crop_model = ClassifierNet(num_classes=len(classes), input_channels=3)
    crop_model.load_state_dict(torch.load(crop_model_path, map_location=torch.device('cpu')))
    crop_model.eval()


    section_images = sections_from_dir(pred_input_dir)

    for section in section_images:
        print(f'Processing section: {section}')
        yolo_results = model(section_images[section])

        crop_model_results = []
        for image in yolo_results:
            crop_model_results_image = []
            for xyxy in image.boxes.xyxy:
                left, top, right, bottom = tuple(xyxy.tolist())
                cropped_img = image.orig_img[int(top):int(bottom), int(left):int(right)] # Crop the image

                cropped_img_resized = cv2.cvtColor(cv2.resize(cropped_img, (128, 128)),
                                                   cv2.COLOR_RGB2BGR) # resize the image

                transform = transforms.Compose([
                    transforms.ToTensor(),  # Convert image to a PyTorch tensor (C x H x W)
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    # Normalize with ImageNet mean/std
                ])

                cropped_img_resized = transform(cropped_img_resized)

                cropped_img_tensor = cropped_img_resized.float().unsqueeze(0)
                with torch.no_grad():
                    crop_prediction = crop_model(cropped_img_tensor)
                crop_model_results_image.append(crop_prediction)
            crop_model_results.append(crop_model_results_image)

        create_gif_and_display(yolo_results, crop_model_results, output_dir = output_dir)

        pred_output_dir = 'pred/bbox3d_body'
        os.makedirs(os.path.join(output_dir, pred_output_dir) ,exist_ok=True)
        create_aidrivemetrics_output_pred(section_images[section], yolo_results, os.path.join(output_dir, pred_output_dir),
                                          crop_model_outputs=crop_model_results)

        gt_output_dir = 'gt/bbox3d_body'
        os.makedirs(os.path.join(output_dir, gt_output_dir), exist_ok=True)
        create_aidrivemetrics_output_gt(section, section_images[section], gt_input_dir, os.path.join(output_dir, gt_output_dir),
                                        img_sizes = [x.orig_img.shape for x in yolo_results],
                                        event_based = create_event_based)
