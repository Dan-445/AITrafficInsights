#speed detection of vehicles across multiple junctions
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2
import pandas as pd
from datetime import datetime, timedelta
import time
import copy
from openpyxl import load_workbook
import torch
# from google.colab.patches import cv2_imshow
model = YOLO("yolov8n.pt")
names = model.model.names



# Video Capture
cap = cv2.VideoCapture("/content/drive/Othercomputers/My_Computer/Dean/114.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("/content/drive/MyDrive/nine3/one1four.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# down line points
line_pts = [(300, 300), (700, 150)]
line_pts2 = [(700, 150), (800, 100)]

start_time = time.time()
update_interval = 900
age_brackets = [
    (1, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39),
    (40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69),
    (70, 74), (75, 998)
]

bracket_counts_init = {
    "up": {
            '1-14': 0, '15-19': 0, '20-24': 0, '25-29': 0, '30-34': 0, '35-39': 0, '40-44': 0, '45-49': 0,'50-54': 0, '55-59': 0, '60-64': 0, '65-69': 0, '70-74': 0, '75-998': 0
    },
    "down": {
            '1-14': 0, '15-19': 0, '20-24': 0, '25-29': 0, '30-34': 0, '35-39': 0, '40-44': 0, '45-49': 0,'50-54': 0, '55-59': 0, '60-64': 0, '65-69': 0, '70-74': 0, '75-998': 0
    },
}

bracket_counts_up = []
bracket_counts_down = []

prevKeyRef = {}

# Define the function to save data to Excel with each dictionary item as a row, including timestamps
def save_to_excel_with_timestamps(row1, row2):
    global bracket_counts_up, bracket_counts_down, start_time

    # print(row1, row2, 'checking rows in print method')

    bracket_counts_up.append(row1)
    bracket_counts_down.append(row2)
    file_path = '/content/drive/MyDrive/nine3/one1four.xlsx'

    # Convert list of dictionaries to DataFrame
    upDataFrame = pd.DataFrame(bracket_counts_up)
    downDataFrame = pd.DataFrame(bracket_counts_down)

    # Save DataFrame to Excel
    with pd.ExcelWriter(file_path) as writer:
        upDataFrame.to_excel(writer, sheet_name='up', index=False)
        downDataFrame.to_excel(writer, sheet_name='down', index=False)

    start_time = time.time()
    return file_path

# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator(reg_pts=line_pts,
                   names=names,
                   view_img=True)
# speed_obj.set_args(reg_pts=line_pts,
#                    names=names,
#                    view_img=True)

speed_obj2 = speed_estimation.SpeedEstimator(reg_pts=line_pts2,
                    names=names,
                    view_img=True)  # Another SpeedEstimator object for the new line
# speed_obj2.set_args(reg_pts=line_pts2,
#                     names=names,
#                     view_img=True)

count = 0
internalData = copy.deepcopy(bracket_counts_init)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)
    trackIds = tracks[0].boxes.id
    if trackIds is not None:
        for trackId in trackIds:
            if int(trackId) not in prevKeyRef:
                sheetName = None
                key = int(trackId)
                value = None
                if(speed_obj.dist_data.get(key)):
                    sheetName= 'down'
                    value = speed_obj.dist_data.get(key)
                if(speed_obj2.dist_data.get(key)):
                    sheetName= 'up'
                    value = speed_obj2.dist_data.get(key)
                if value is not None:
                    prevKeyRef[key] = key
                    for start, end in age_brackets:
                        if start <= value <= end:
                            internalData[sheetName][f"{start}-{end}"] += 1
                            count += 1

    current_time = time.time()
    print(current_time, start_time,  current_time - start_time, update_interval, count, 'hhmmmmmm')
    if current_time - start_time >= update_interval:
        count = 0
        cTime = {'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))}

        # print(internalData, 'internalData is here')
        internalDataToProcess = copy.deepcopy(internalData)

        row1 = copy.copy(cTime)
        row1Data = copy.copy(internalDataToProcess['up'])
        row1.update(row1Data)

        row2 = copy.copy(cTime)
        row2Data = copy.copy(internalDataToProcess['down'])
        row2.update(row2Data)

        # print(row1, row2, 'rows count inner')

        internalData = copy.deepcopy(bracket_counts_init)
        # print(internalData, 'internalData after reset')

        save_to_excel_with_timestamps(row1, row2)

    im0 = speed_obj.estimate_speed(im0, tracks)
    im0 = speed_obj2.estimate_speed(im0, tracks)
    video_writer.write(im0)
    # cv2.line(im0,(700, 150), (800, 100), color = (0, 255, 0), thickness = 9)
    # cv2_imshow(im0)


    print('frame =', cap.get(cv2.CAP_PROP_POS_FRAMES))
    # if cap.get(cv2.CAP_PROP_POS_FRAMES) >= 1:
    #  print('frames reached limit')
    #  break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
