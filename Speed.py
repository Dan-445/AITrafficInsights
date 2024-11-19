from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime
import time
import copy
from openpyxl import load_workbook
import torch

# Initialize YOLO model
model = YOLO("yolo11n.pt")  # Ensure 'yolov8n.pt' is in your working directory or provide the correct path
names = model.model.names  # Dictionary mapping class IDs to class names

# Video Capture
cap = cv2.VideoCapture("vid.mp4")  # Replace with your video path
if not cap.isOpened():
    raise IOError("Error opening video file")

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                       cv2.CAP_PROP_FRAME_HEIGHT, 
                                       cv2.CAP_PROP_FPS))
print(f"Video Width: {w}, Height: {h}, FPS: {fps}")

# Video writer
video_writer = cv2.VideoWriter("one1four.mp4",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Define line points for speed estimation and direction detection
# These lines represent the regions where crossing will be detected
# line_pts for 'down' direction, line_pts2 for 'up' direction
line_pts = [(745, 321), (949, 348)]    # Downward crossing line
line_pts2 = [(734, 328), (615, 293)]   # Upward crossing line

start_time = time.time()
update_interval = 200  # 15 minutes in seconds

# Define speed brackets (age_brackets might be better named speed_brackets)
speed_brackets = [
    (1, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39),
    (40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69),
    (70, 74), (75, 998)
]

bracket_counts_init = {
    "up": {f"{start}-{end}": 0 for start, end in speed_brackets},
    "down": {f"{start}-{end}": 0 for start, end in speed_brackets},
}

bracket_counts_up = []
bracket_counts_down = []
prevKeyRef = {}

# Define the function to save data to Excel
def save_to_excel_with_timestamps(row1, row2):
    global bracket_counts_up, bracket_counts_down, start_time

    bracket_counts_up.append(row1)
    bracket_counts_down.append(row2)
    file_path = 'one1four.xlsx'

    # Convert list of dictionaries to DataFrame
    upDataFrame = pd.DataFrame(bracket_counts_up)
    downDataFrame = pd.DataFrame(bracket_counts_down)

    # Save DataFrame to Excel
    with pd.ExcelWriter(file_path) as writer:
        upDataFrame.to_excel(writer, sheet_name='up', index=False)
        downDataFrame.to_excel(writer, sheet_name='down', index=False)

    start_time = time.time()
    return file_path

# Calibration factor (meters per pixel). **You need to determine this based on your camera setup.**
# Example: If 100 pixels = 5 meters, then meters_per_pixel = 0.05
meters_per_pixel = 0.05  # Update this value accordingly

# Define vehicle class IDs based on the model's class names
vehicle_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat']  # Update as needed
vehicle_class_ids = [cls_id for cls_id, cls_name in names.items() if cls_name in vehicle_classes]

# Initialize tracking data structures
track_positions = {}      # Stores the last positions of each track ID
track_speeds = {}         # Stores the calculated speed of each track ID
track_directions = {}     # Stores the direction ('up' or 'down') of each track ID
track_counted = {}        # Stores whether a track has been counted for 'up' or 'down'

count = 0
internalData = copy.deepcopy(bracket_counts_init)

# Create a named window for display (optional)
cv2.namedWindow("Speed Detection", cv2.WINDOW_NORMAL)

# Define font parameters for labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 2

# Function to determine which side of a line a point is on
def get_side(p1, p2, point):
    """
    Determine the side of the point relative to the line formed by p1 and p2.
    Returns a positive value if the point is on one side, negative on the other, and zero if on the line.
    """
    return (p2[0] - p1[0])*(point[1] - p1[1]) - (p2[1] - p1[1])*(point[0] - p1[0])

# Main processing loop
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking
    results = model.track(im0, persist=True, show=False)
    trackIds = results[0].boxes.id  # List of track IDs in the current frame

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if trackIds is not None:
        for track in results[0].boxes:
            cls_id = int(track.cls)  # Class ID
            if cls_id not in vehicle_class_ids:
                continue  # Skip non-vehicle classes

            trackId = int(track.id)
            bbox_tensor = track.xyxy  # Bounding box tensor [1, 4]

            try:
                # Convert tensor to list
                bbox = bbox_tensor.cpu().numpy().flatten().tolist()

                # Validate bbox length
                if len(bbox) != 4:
                    raise ValueError(f"Expected 4 elements in bbox, got {len(bbox)}")

                x1, y1, x2, y2 = map(int, bbox)  # Convert coordinates to integers
                centroid = (int((x1 + x2)/2), int((y1 + y2)/2))

            except Exception as e:
                print(f"Error processing Track ID: {trackId}: {e}. Skipping...")
                continue

            # Update track_positions
            if trackId not in track_positions:
                track_positions[trackId] = []
            track_positions[trackId].append((current_frame, centroid))

            # Keep only the last two positions to calculate speed
            if len(track_positions[trackId]) > 2:
                track_positions[trackId].pop(0)

            # Initialize tracking flags
            if trackId not in track_counted:
                track_counted[trackId] = {'up': False, 'down': False}

            # Calculate speed and detect crossing if there are at least two positions
            if len(track_positions[trackId]) == 2:
                frame_diff = track_positions[trackId][1][0] - track_positions[trackId][0][0]
                if frame_diff > 0:
                    # Calculate pixel distance
                    pixel_distance = ((track_positions[trackId][1][1][0] - track_positions[trackId][0][1][0]) ** 2 + 
                                      (track_positions[trackId][1][1][1] - track_positions[trackId][0][1][1]) ** 2) ** 0.5
                    # Calculate speed in meters per second
                    speed_mps = (pixel_distance * meters_per_pixel * fps) / frame_diff
                    # Convert to km/h
                    speed_kmh = speed_mps * 3.6
                    track_speeds[trackId] = speed_kmh

                    # Determine direction based on crossing lines
                    # Get previous and current centroid positions
                    prev_centroid = track_positions[trackId][0][1]
                    curr_centroid = track_positions[trackId][1][1]

                    # Check crossing for 'down' direction
                    side_prev_down = get_side(line_pts[0], line_pts[1], prev_centroid)
                    side_curr_down = get_side(line_pts[0], line_pts[1], curr_centroid)

                    if side_prev_down * side_curr_down < 0:
                        # Crossing detected
                        if not track_counted[trackId]['down']:
                            direction = 'down'
                            track_counted[trackId]['down'] = True  # Mark as counted

                            # Categorize speed into speed_brackets
                            for start, end in speed_brackets:
                                if start <= speed_kmh <= end:
                                    internalData['down'][f"{start}-{end}"] += 1
                                    count += 1
                                    break

                            # Prepare label with background
                            label = f"ID:{trackId} {speed_kmh:.2f} Mp/h {direction}"
                            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                            text_x, text_y = x1, y1 - 10  # Position above the bounding box

                            # Ensure label doesn't go out of frame
                            text_x = max(text_x, 0)
                            text_y = max(text_y, text_height + baseline)

                            # Calculate the position for the background rectangle
                            rect_top_left = (text_x, text_y - text_height - baseline)
                            rect_bottom_right = (text_x + text_width, text_y + baseline)

                            # Choose background color based on direction
                            bg_color = (0, 0, 255)  # Red for 'down'

                            # Draw the filled rectangle
                            cv2.rectangle(im0, rect_top_left, rect_bottom_right, bg_color, cv2.FILLED)

                            # Draw the text over the rectangle
                            cv2.putText(im0, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                    # Check crossing for 'up' direction
                    side_prev_up = get_side(line_pts2[0], line_pts2[1], prev_centroid)
                    side_curr_up = get_side(line_pts2[0], line_pts2[1], curr_centroid)

                    if side_prev_up * side_curr_up < 0:
                        # Crossing detected
                        if not track_counted[trackId]['up']:
                            direction = 'up'
                            track_counted[trackId]['up'] = True  # Mark as counted

                            # Categorize speed into speed_brackets
                            for start, end in speed_brackets:
                                if start <= speed_kmh <= end:
                                    internalData['up'][f"{start}-{end}"] += 1
                                    count += 1
                                    break

                            # Prepare label with background
                            label = f"ID:{trackId} {speed_kmh:.2f} Mp/h {direction}"
                            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                            text_x, text_y = x1, y1 - 10  # Position above the bounding box

                            # Ensure label doesn't go out of frame
                            text_x = max(text_x, 0)
                            text_y = max(text_y, text_height + baseline)

                            # Calculate the position for the background rectangle
                            rect_top_left = (text_x, text_y - text_height - baseline)
                            rect_bottom_right = (text_x + text_width, text_y + baseline)

                            # Choose background color based on direction
                            bg_color = (0, 255, 0)  # Green for 'up'

                            # Draw the filled rectangle
                            cv2.rectangle(im0, rect_top_left, rect_bottom_right, bg_color, cv2.FILLED)

                            # Draw the text over the rectangle
                            cv2.putText(im0, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Draw the region lines on the frame (optional)
    cv2.line(im0, line_pts[0], line_pts[1], color=(0, 0, 255), thickness=2)   # Red line for 'down' direction
    cv2.line(im0, line_pts2[0], line_pts2[1], color=(0, 255, 0), thickness=2)  # Green line for 'up' direction

    # Draw bounding boxes around detected objects with labels
    if trackIds is not None:
        for track in results[0].boxes:
            cls_id = int(track.cls)
            if cls_id not in vehicle_class_ids:
                continue

            trackId = int(track.id)
            bbox_tensor = track.xyxy
            try:
                bbox = bbox_tensor.cpu().numpy().flatten().tolist()
                if len(bbox) != 4:
                    raise ValueError(f"Expected 4 elements in bbox, got {len(bbox)}")
                x1, y1, x2, y2 = map(int, bbox)
            except Exception as e:
                print(f"Error processing Track ID: {trackId}: {e}. Skipping...")
                continue

            # Define color based on direction
            box_color = (0, 255, 0) if track_directions.get(trackId, 'up') == 'up' else (0, 0, 255)
            cv2.rectangle(im0, (x1, y1), (x2, y2), box_color, 2)

            # Optionally, put class name on the bounding box
            class_name = names.get(cls_id, 'Unknown')
            class_label = class_name
            (class_width, class_height), class_baseline = cv2.getTextSize(class_label, font, font_scale, thickness)
            class_text_x, class_text_y = x1, y1 - 25  # Position above the speed label

            # Calculate the position for the class background rectangle
            class_rect_top_left = (class_text_x, class_text_y - class_height - class_baseline)
            class_rect_bottom_right = (class_text_x + class_width, class_text_y + class_baseline)

            # Draw the filled rectangle for class label
            cv2.rectangle(im0, class_rect_top_left, class_rect_bottom_right, box_color, cv2.FILLED)

            # Draw the class name text
            cv2.putText(im0, class_label, (class_text_x, class_text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Display the frame with bounding boxes and annotations
    cv2.imshow("Speed Detection", im0)

    # Write the frame to the output video
    video_writer.write(im0)

    # Print the processed frame number
    print('Processed frame:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break

    # Periodically save data to Excel
    current_time = time.time()
    if current_time - start_time >= update_interval:
        count = 0
        cTime = {'Timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))}

        internalDataToProcess = copy.deepcopy(internalData)

        row1 = copy.copy(cTime)
        row1Data = copy.copy(internalDataToProcess['up'])
        row1.update(row1Data)

        row2 = copy.copy(cTime)
        row2Data = copy.copy(internalDataToProcess['down'])
        row2.update(row2Data)

        internalData = copy.deepcopy(bracket_counts_init)
        save_to_excel_with_timestamps(row1, row2)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
