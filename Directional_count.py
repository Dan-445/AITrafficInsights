import os
import cv2
import torch
from ultralytics import YOLO
import pandas as pd
import time
from datetime import datetime
import copy
import numpy as np


DISPLAY = True 


counting_lines = {
    "nb": {  # Northbound
        "incoming": [(215, 545), (590, 610)],  # Define accurately or set to [(0,0), (0,0)] if not used
        "left": [(590, 610), (780, 635)],
        "thru": [(784, 642), (980, 680)],
        "right": [(984, 681), (1205, 715)]  # Adjust as per video
    },
    "sb": {  # Southbound
        "incoming": [(1262, 376), (1059, 356)],  # Placeholder if not needed
        "left": [(1059, 356), (954, 346)],
        "thru": [(954, 346), (861, 333)],
        "right": [(861, 333), (726, 323)]  # Adjust as per video
    },
    "wb": {  # Westbound
        "incoming": [(1278, 697), (1292, 529)],  # Placeholder if not needed
        "left": [(1292, 529), (1296, 476)],
        "thru": [(1296, 476), (1299, 435)],
        "right": [(1299, 435), (1301, 389)]  # Adjust as per video
    },
    "eb": {  # Eastbound
        "incoming": [(620, 329), (505, 386)],  # Placeholder if not needed
        "left": [(505, 386), (439, 424)],
        "thru": [(439, 424), (354, 465)],
        "right": [(354, 465), (248, 519)]  # Adjust as per video
    }
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("best.pt")  h
model.to(device)

# Define target classes and their colors
target_classes = ['car', 'truck', 'bus']
class_colors = {
    'car': (0, 255, 0),      # Green
    'truck': (255, 0, 0),    # Blue
    'bus': (0, 0, 255)        # Red
}

direction_colors = {
    "nb": (255, 0, 0),    # Blue
    "sb": (0, 198, 255),  # Yellow
    "wb": (0, 165, 255),  # Orange
    "eb": (255, 0, 255)   # Magenta
}

# Convert class names to a list
all_class_names = list(model.names.values())

# Filter class indices for target classes
target_class_indices = [i for i, name in enumerate(all_class_names) if name in target_classes]

# Create a mapping from class index to class name for target classes
target_class_mapping = {i: name for i, name in enumerate(all_class_names) if name in target_classes}

vehicle_data = {
    direction: {
        cls: {
            "incoming": 0,
            "left": 0,
            "thru": 0,
            "right": 0,
            "Total": 0
        } for cls in target_classes
    } for direction in counting_lines.keys()
}

start_time = time.time()


def draw_label_with_background(frame, text, position, font, font_scale, font_thickness, text_color, bg_color):
    """
    Draws text with a background rectangle for better readability.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position
    # Draw rectangle
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bg_color, cv2.FILLED)
    # Put text over the rectangle
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

def do_lines_intersect(line1, line2):
    """
    Determines if two lines (each defined by two points) intersect.

    Parameters:
        line1: List of two tuples [(x1, y1), (x2, y2)]
        line2: List of two tuples [(x3, y3), (x4, y4)]

    Returns:
        True if the lines intersect, False otherwise.
    """
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    A, B = line1
    C, D = line2

    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def process_data(new_data):
    """
    Aggregates counts from all counters and appends them to vehicle_data.
    """
    global vehicle_data
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for direction, classes in new_data.items():
        for cls, counts in classes.items():
            for lane, count in counts.items():
                vehicle_data[direction][cls][lane] += count

def save_to_excel_with_timestamps(new_data):
    """
    Saves the aggregated vehicle_data into an Excel file with separate sheets for each direction.
    """
    global start_time

    process_data(new_data)
    print("Creating Excel Rows:", vehicle_data)
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        for direction in vehicle_data:
            data = []
            for cls in vehicle_data[direction]:
                entry = vehicle_data[direction][cls]
                data.append({
                    "Class": cls,
                    "Incoming": entry["incoming"],
                    "Left": entry["left"],
                    "Thru": entry["thru"],
                    "Right": entry["right"],
                    "Total": entry["Total"]
                })
            df_direction = pd.DataFrame(data)
            df_direction.to_excel(writer, sheet_name=direction.upper(), index=False)
    
    # Reset start_time for the next interval
    start_time = time.time()

class CentroidTracker:
    def __init__(self, max_disappeared=0, max_distance=50):
        # Initialize the next unique object ID along with two ordered dictionaries
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        
        # Store the number of maximum consecutive frames a given object is allowed to be marked as "disappeared"
        self.max_disappeared = max_disappeared
        
        # Store the maximum distance between centroids to associate them
        self.max_distance = max_distance
    
    def register(self, centroid):
        # Register a new object with a unique ID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        # Deregister an object ID
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        # Check if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # Mark existing objects as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                # Deregister if exceeded maximum allowed disappeared frames
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            
            # Return early as there are no detections
            return self.objects
        
        # Initialize an array of input centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)
        
        # If no existing objects, register all input centroids
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        
        else:
            # Grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            # Compute distance between each pair of object centroids and input centroids
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - input_centroids, axis=2)
            
            # Find the smallest value in each row and then sort the row indices based on their minimum values
            rows = D.min(axis=1).argsort()
            
            # Similarly, find the smallest value in each column and sort based on their minimum values
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of already examined rows and columns
            usedRows = set()
            usedCols = set()
            
            # Iterate over the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # Ignore if row or column is already used
                if row in usedRows or col in usedCols:
                    continue
                
                # If distance is greater than maximum, ignore
                if D[row, col] > self.max_distance:
                    continue
                
                # Otherwise, update the centroid and reset disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                
                # Mark row and column as used
                usedRows.add(row)
                usedCols.add(col)
            
            # Compute unused rows and columns
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # Handle disappeared objects
            if D.shape[0] >= D.shape[1]:
                # More existing objects than input centroids
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                # More input centroids than existing objects
                for col in unusedCols:
                    self.register(input_centroids[col])
        
        return self.objects

def process_video(video_path, output_path):
    """
    Processes the video, tracks objects crossing defined lines, counts them, and writes annotated frames to output.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start processing from frame 0

    if not cap.isOpened():
        print(f"Error reading video file {video_path}. Skipping...")
        return

    # Initialize Centroid Tracker
    ct = CentroidTracker(max_disappeared=40, max_distance=50)
    trackableObjects = {}

    # Initialize video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (w, h)
    )

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            print("Reached end of video or failed to read the frame.")
            break

        frame_count += 1

        # Run YOLO model on the frame
        results = model(frame, device=device)
        detections = results[0].boxes  # Adjust based on actual Ultralytics YOLO API

        rects = []

        # Process detections and draw bounding boxes and labels
        for box in detections:
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id not in target_class_mapping:
                continue  # Skip non-target classes
            cls_name = target_class_mapping[cls_id]
            confidence = box.conf[0].cpu().numpy()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            color = class_colors.get(cls_name, (0, 255, 0))  # Default to green if not found

            # Append bounding box to rects list for tracking
            rects.append((int(x1), int(y1), int(x2), int(y2)))

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Prepare label with background
            label = f"{cls_name} {confidence:.2f}"
            draw_label_with_background(
                frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2,
                (255, 255, 255),  # White text
                color  # Background color same as bounding box
            )

        # Update Centroid Tracker with detected bounding boxes
        objects = ct.update(rects)

        # Iterate over tracked objects
        for (objectID, centroid) in objects.items():
            # Initialize trackable object if not present
            to = trackableObjects.get(objectID, {"centroids": [], "counted": False})
            to["centroids"].append(centroid)
            trackableObjects[objectID] = to

            # Draw the object ID near its centroid
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

            # Check for line crossing only if not already counted
            if not to["counted"]:
                for direction in counting_lines:
                    for lane in counting_lines[direction]:
                        line = counting_lines[direction][lane]
                        if line == [(0,0), (0,0)]:
                            continue  # Skip if line is not defined

                        # Previous centroid
                        if len(to["centroids"]) >= 2:
                            prev_centroid = to["centroids"][-2]
                            curr_centroid = to["centroids"][-1]

                            # Define the movement line segment
                            movement_line = [tuple(prev_centroid), tuple(curr_centroid)]
                            counting_line = [tuple(line[0]), tuple(line[1])]

                            # Check for intersection
                            if do_lines_intersect(movement_line, counting_line):
                                # Increment the count for the specific lane and class
                                vehicle_data[direction][cls_name][lane] += 1
                                vehicle_data[direction][cls_name]["Total"] += 1

                                # Mark this object as counted
                                to["counted"] = True
                                trackableObjects[objectID] = to

                                # Debugging information
                                print(f"Object ID {objectID} ({cls_name}) crossed {direction.upper()} {lane} lane. Total {lane}: {vehicle_data[direction][cls_name][lane]}")

                                break
                    # If already counted, no need to check other directions
                    if to["counted"]:
                        break

        # Draw counting lines with labels
        for direction in counting_lines:
            for lane in counting_lines[direction]:
                line = counting_lines[direction][lane]
                if line == [(0,0), (0,0)]:
                    continue  # Skip if line is not defined
                cv2.line(frame, line[0], line[1], (255, 255, 0), 2)  # Cyan color for lines
                # Put label near the line
                midpoint = ((line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2)
                label = f"{direction.upper()} {lane.capitalize()}"
                draw_label_with_background(
                    frame,
                    label,
                    (midpoint[0], midpoint[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2,
                    (0, 0, 0),  # Black text
                    (255, 255, 0)  # Cyan background
                )

        # Draw the counts on the frame (4 boxes for 4 directions)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (255, 255, 255)  # White text

        # Define box positions for each direction
        box_positions = {
            "nb": ((10, 10), (0, 0)),
            "sb": ((370, 10), (0, 0)),
            "wb": ((700, 10), (0, 0)),
            "eb": ((1120, 10), (0, 0))
        }

        # Define text positions corresponding to box positions
        text_positions = {
            "nb": (box_positions["nb"][0][0] + 10, box_positions["nb"][0][1] + 30),
            "sb": (box_positions["sb"][0][0] + 10, box_positions["sb"][0][1] + 30),
            "wb": (box_positions["wb"][0][0] + 10, box_positions["wb"][0][1] + 30),
            "eb": (box_positions["eb"][0][0] + 10, box_positions["eb"][0][1] + 30)
        }

        for direction in ['nb', 'sb', 'wb', 'eb']:
            counts = vehicle_data.get(direction, {})
            if not counts:
                continue
            # Aggregate counts across all classes for the direction
            total_in = sum([counts[cls]["incoming"] for cls in target_classes])
            total_left = sum([counts[cls]["left"] for cls in target_classes])
            total_thru = sum([counts[cls]["thru"] for cls in target_classes])
            total_right = sum([counts[cls]["right"] for cls in target_classes])
            total = sum([counts[cls]["Total"] for cls in target_classes])

            # Prepare text
            text = f"{direction.upper()}: Inc={total_in} Left={total_left} Thru={total_thru} Right={total_right} Total={total}"

            # Draw filled rectangle for background
            cv2.rectangle(
                frame,
                box_positions[direction][0],
                box_positions[direction][1],
                direction_colors[direction],
                cv2.FILLED
            )

            # Put text with background
            draw_label_with_background(
                frame,
                text,
                text_positions[direction],
                font,
                font_scale,
                font_thickness,
                text_color,
                direction_colors[direction]
            )

        # Optionally display the frame with annotations
        if DISPLAY:
            cv2.imshow('Annotated Frame', frame)
            # Press 'q' to exit the display window early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Early termination requested. Exiting...")
                break

        # Write the annotated frame to the output video
        video_writer.write(frame)

        # Debugging info
        print(f"Processed Frame {frame_count}/{total_frames} of {video_path}")

    # After processing all frames, save the counts to Excel
    new_data = copy.deepcopy(vehicle_data)
    save_to_excel_with_timestamps(new_data)

    # Release video resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


# Directory containing the videos
video_dir = "videos"  # Ensure this directory exists and contains your video files
output_excel_path = 'traffic_count_results.xlsx'

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each video in the directory
for video_file in os.listdir(video_dir):
    if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add or remove extensions as needed
        print(f"Processing video: {video_file}")
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"output_{os.path.splitext(video_file)[0]}.mp4")
        print(f"Output will be saved to: {output_path}")
        process_video(video_path, output_path)
        print(f"Finished processing video: {video_file}\n")
