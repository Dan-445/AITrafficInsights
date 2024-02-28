import cv2
from sort import *
import math
import numpy as np
from ultralytics import YOLO
import cvzone
import pandas as pd
import time

cap = cv2.VideoCapture('fmail/Standard_SCUEW9_2024-02-14_0500.013.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
out = cv2.VideoWriter('fmail/443SSout2024_0208_070827_042A.MP4', fourcc, fps, (frame_width, frame_height))

# model = YOLO('/Users/dan/tracking/dean/yolov8-multiple-vehicle-class/yen/besttrack2.pt')
model = YOLO('/Users/dan/tracking/dean/yolov8-multiple-vehicle-class/yen/azbest.pt')

classnames = []
with open('/Users/dan/tracking/dean/yolov8-multiple-vehicle-class/yen/classes.txt', 'r') as f:
    classnames = f.read().splitlines()

tracker = Sort(max_age=20)
outgoing = [64, 482, 631, 255]
left = [64, 482, 104, 214]
right = [183, 151, 335, 108]
incoming = [335, 108, 448, 160]  # New incoming line
counter_car_outgoing = {}
counter_bus_outgoing = {}
counter_truck_outgoing = {}
counter_car_left = {}
counter_bus_left = {}
counter_truck_left = {}
counter_car_right = {}
counter_bus_right = {}
counter_truck_right = {}
counter_car_incoming = {}  # Counter for cars incoming
counter_bus_incoming = {}  # Counter for buses incoming
counter_truck_incoming = {}  # Counter for trucks incoming

data = []

start_time = time.time()
interval = 600  # 10 minutes in seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = np.empty((0, 5))
    result = model(frame, stream=1)
    for info in result:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            classindex = box.cls[0]
            conf = math.ceil(conf * 100)
            classindex = int(classindex)
            objectdetect = classnames[classindex]

            if conf > 19:
                if objectdetect == ' 2 Axel Suv':
                    counter_outgoing = counter_car_outgoing
                    counter_left = counter_car_left
                    counter_right = counter_car_right
                    counter_incoming = counter_car_incoming
                elif objectdetect == 'Car':
                    counter_outgoing = counter_bus_outgoing
                    counter_left = counter_bus_left
                    counter_right = counter_bus_right
                    counter_incoming = counter_bus_incoming
                elif objectdetect == ' 2 Axel Truck':
                    counter_outgoing = counter_truck_outgoing
                    counter_left = counter_truck_left
                    counter_right = counter_truck_right
                    counter_incoming = counter_truck_incoming
                    
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                new_detections = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, new_detections))

                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                if outgoing[0] < cx < outgoing[2] and outgoing[1] - 20 < cy < outgoing[1] + 20:
                    if id not in counter_outgoing:
                        counter_outgoing[id] = time.time()
                elif left[0] < cx < left[2] and left[1] - 20 < cy < left[1] + 20:
                    if id not in counter_left:
                        counter_left[id] = time.time()
                elif right[0] < cx < right[2] and right[1] - 20 < cy < right[1] + 20:
                    if id not in counter_right:
                        counter_right[id] = time.time()
                elif incoming[0] < cx < incoming[2] and incoming[1] - 20 < cy < incoming[1] + 20:  # Check for incoming line
                    if id not in counter_incoming:
                        counter_incoming[id] = time.time()

    track_result = tracker.update(detections)
    cv2.line(frame, (outgoing[0], outgoing[1]), (outgoing[2], outgoing[3]), (0, 255, 255), 1)
    cv2.line(frame, (left[0], left[1]), (left[2], left[3]), (255, 0, 0), 1)
    cv2.line(frame, (right[0], right[1]), (right[2], right[3]), (0, 0, 255), 1)
    cv2.line(frame, (incoming[0], incoming[1]), (incoming[2], incoming[3]), (255, 255, 0), 1)  # Draw incoming line

    for results in track_result:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cvzone.putTextRect(frame, f'{id}', [x1 + 8, y1 - 12], thickness=1, scale=0.5)

        if outgoing[0] < cx < outgoing[2] and outgoing[1] - 20 < cy < outgoing[1] + 20:
            cv2.line(frame, (outgoing[0], outgoing[1]), (outgoing[2], outgoing[3]), (0, 0, 255), 1)
        elif incoming[0] < cx < incoming[2] and incoming[1] - 20 < cy < incoming[1] + 20:  # Check if object crosses incoming line
            cv2.line(frame, (incoming[0], incoming[1]), (incoming[2], incoming[3]), (255, 0, 255), 1)

    current_time = time.time()
    if current_time - start_time >= interval:
        # Update DataFrame
        results_data = {
            '2 Axel Suv outgoing': [len(counter_car_outgoing)],
            'cars outgoing': [len(counter_bus_outgoing)],
            '2 Axel Truck outgoing': [len(counter_truck_outgoing)],
            '2 Axel Suv left': [len(counter_car_left)],
            'cars left': [len(counter_bus_left)],
            '2 Axel Truck left': [len(counter_truck_left)],
            '2 Axel Suv right': [len(counter_car_right)],
            'cars right': [len(counter_bus_right)],
            '2 Axel Truck right': [len(counter_truck_right)],
            '2 Axel Suv incoming': [len(counter_car_incoming)],  # Add incoming counts to DataFrame
            'cars incoming': [len(counter_bus_incoming)],
            '2 Axel Truck incoming': [len(counter_truck_incoming)]
        }
        results_df = pd.DataFrame(results_data)
        results_df.to_excel(f'fmail/443.MP4results_{int(start_time)}.xlsx', index=False)
        
        # Reset counters
        counter_car_outgoing = {}
        counter_bus_outgoing = {}
        counter_truck_outgoing = {}
        counter_car_left = {}
        counter_bus_left = {}
        counter_truck_left = {}
        counter_car_right = {}
        counter_bus_right = {}
        counter_truck_right = {}
        counter_car_incoming = {}  # Reset incoming counters
        counter_bus_incoming = {}
        counter_truck_incoming = {}

        start_time = current_time

    out.write(frame)  # Write the frame to the video file

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()
