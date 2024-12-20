
import hydra
import math
import torch
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
import csv

import pandas as pd
from datetime import datetime

data_deque = {}

deepsort = None

speed_line_queue = {}

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

detected_vehicle_ids = set()
vehicle_counts = {'vehicle_counts': 0}

# Create a CSV file for output
csv_file = open('detected_vehicle_count.csv', 'w', newline='')  # Open a CSV file in write mode
csv_writer = csv.writer(csv_file)  # Create a CSV writer object
csv_writer.writerow(['Class', 'ID', 'Orientation', 'speed','time'])  # Write the header row to the CSV file


def estimatespeed(Location1, Location2):
    # Euclidean Distance Formula
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining thr pixels per meter
    ppm = 8
    d_meters = d_pixel / ppm
    time_constant = 15 * 0.5
    # distance = speed/time
    speed = d_meters * time_constant

    return int(speed)


def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    return img


def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=1)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 1, 1)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=1)


def get_direction(point1, point2):
    # Calculate the orientation in degrees
    point2 = (point2[0] + 0.1, point2[1] + 0.1)
    orientation = np.arctan2(point2[1] - point1[1], point2[0] - point1[0]) * 180 / np.pi
    if -45 <= orientation <= 45:
        direction_str = "east"
    elif 45 < orientation <= 135:
        direction_str = "north"
    elif 135 < orientation <= 225:
        direction_str = "west"
    else:
        direction_str = "south"

    return direction_str


def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    # cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2 + x1) / 2), int((y2 + y2) / 2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
            speed_line_queue[id] = []

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]

        label = '%s' % (obj_name)
        # Check if the vehicle ID is new
        detection_id = id
        if detection_id not in detected_vehicle_ids:
            detected_vehicle_ids.add(detection_id)
            if obj_name in ['car','truck','bus','motorcycle','bicycle']:
              vehicle_counts['vehicle_counts'] += 1

        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        font_thickness = 2
        font_color = (255,255,255)
        text_x = 20
        text_y = 40
        line_height = 45  # Adjust this value to control the vertical spacing

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            object_speed = estimatespeed(data_deque[id][1], data_deque[id][0])
            if object_speed >0:
              #current_time = datetime.now()
              current_time = datetime.now()
              formatted_time = current_time.strftime("%H:%M:%S")
              csv_writer.writerow([obj_name, detection_id, direction, object_speed,formatted_time ])
              speed_line_queue[id].append(object_speed)

        try:
            label = label + " " + str(sum(speed_line_queue[id]) // len(speed_line_queue[id])) + "km/h" + " " + direction
        except:
            pass
        UI_box(box, img, label=label, color=color, line_thickness=2)
    return img



class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1.5
        font_thickness = 2
        font_color = (255,255,255)
        text_x = 20
        text_y = 40
        line_height = 45  # Adjust this value to control the vertical spacing
        for vehicle_class, count in vehicle_counts.items():
            class_label = f'{vehicle_class}:{count}'
            cv2.putText(img, class_label, (text_x, text_y), font, font_scale, font_color, font_thickness)
            text_y += line_height
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        vehicle_class_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # Labels for various vehicle classes
        filtered_preds = []  # List to store filtered predictions
        vehicle_class_labels_tensor = torch.tensor(vehicle_class_labels).to(
            self.model.device)  # Move to the same device as the model
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

            # Filter out predictions for vehicle classes
            vehicle_mask = torch.any(pred[:, 5].unsqueeze(0) == vehicle_class_labels_tensor.unsqueeze(1), dim=0)
            vehicle_pred = pred[vehicle_mask]

            if vehicle_pred.shape[0] > 0:
                filtered_preds.append(vehicle_pred)
        return filtered_preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)
        if len(preds) != 0:
            det = preds[idx]
            all_outputs.append(det)
            if len(det) == 0:
                return log_string
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
            # write
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            xywh_bboxs = []
            confs = []
            oids = []
            outputs = []
            for *xyxy, conf, cls in reversed(det):
                x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                xywh_bboxs.append(xywh_obj)
                confs.append([conf.item()])
                oids.append(int(cls))
            xywhs = torch.Tensor(xywh_bboxs)
            confss = torch.Tensor(confs)
            outputs = deepsort.update(xywhs, confss, oids, im0)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -2]
                object_id = outputs[:, -1]
                draw_boxes(im0, bbox_xyxy, self.model.names, object_id, identities)
            return log_string
        return ''

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
