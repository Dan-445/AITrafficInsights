import os
import cv2
from ultralytics import YOLO, solutions
import pandas as pd
import time
from datetime import datetime, timedelta
import copy
# Initialize the YOLO model
model = YOLO("countbest.pt")

# Helper function to calculate combined values and append to the target structure
def process_data(new_data):
    global vehicle_data
    cTime = {'Timestamp': (datetime.fromtimestamp(start_time) + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")}

    for key in ['PV', 'TTST', 'Duals', 'Twins']:
        rightData = new_data[0].get(key, {})
        leftData = new_data[1].get(key, {})
        thruData = new_data[2].get(key, {})
        incommingData = new_data[3].get(key, {})

        combined_right = rightData.get('IN', 0) + rightData.get('OUT', 0)
        combined_left = leftData.get('IN', 0) + leftData.get('OUT', 0)
        combined_thru = thruData.get('IN', 0) + thruData.get('OUT', 0)
        combined_incoming = incommingData.get('IN', 0) + incommingData.get('OUT', 0)

        prev_row = {}
        if len(vehicle_data[key]):
            prev_row = vehicle_data[key][-1]

        prevRowCount = prev_row.get("prevRawCount", {})
        vehicle_data[key].append({
            "TimeStamp": cTime['Timestamp'],
            "right": combined_right - (prevRowCount.get('right', 0)),
            "left": combined_left - (prevRowCount.get('left', 0)),
            "thru": combined_thru - (prevRowCount.get('thru', 0)),
            "incoming": combined_incoming - (prevRowCount.get('incoming', 0)),
            "prevRawCount": {"right": combined_right, "left": combined_left, "thru": combined_thru,"incoming": combined_incoming}
        })

def save_to_excel_with_timestamps(new_data):
    global start_time

    process_data(new_data)
    print("Creating Excel Row", vehicle_data)

    # Convert list of dictionaries to DataFrame
    PV = pd.DataFrame(vehicle_data.get('PV'))
    PV = PV.drop(columns=["prevRawCount"])
    
    TTST = pd.DataFrame(vehicle_data.get('TTST'))
    TTST = TTST.drop(columns=["prevRawCount"])
    
    Duals = pd.DataFrame(vehicle_data.get('Duals'))
    Duals = Duals.drop(columns=["prevRawCount"])

    Twins = pd.DataFrame(vehicle_data.get('Twins'))
    Twins = Twins.drop(columns=["prevRawCount"])

    # Save DataFrame to Excel
    with pd.ExcelWriter(output_excel_path) as writer:
        PV.to_excel(writer, sheet_name='PV', index=False)
        TTST.to_excel(writer, sheet_name='TTST', index=False)
        Duals.to_excel(writer, sheet_name='Duals', index=False)
        Twins.to_excel(writer, sheet_name='Twins', index=False)

    start_time = (datetime.fromtimestamp(start_time) + timedelta(minutes=15)).timestamp()

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 900)

    if not cap.isOpened():
        print(f"Error reading video file {video_path}. Skipping...")
        return
    
    # Init Object Counter
    nb_counter_incomming = solutions.ObjectCounter(
        classes_names=model.names,
        reg_pts=[(638, 174), (755, 138)],
        view_img=False,
        draw_tracks=True,
        line_thickness=1,
    )

    nb_counter_left = solutions.ObjectCounter(
        classes_names=model.names,
        reg_pts=[(509, 220), (584, 191)],
        view_img=False,
        draw_tracks=True,
        line_thickness=1,
    )

    nb_counter_thru = solutions.ObjectCounter(
        classes_names=model.names,
        reg_pts=[(429, 250),(499, 222)],
        view_img=False,
        draw_tracks=True,
        line_thickness=1,
    )

    nb_counter_right = solutions.ObjectCounter(
        classes_names=model.names,
        reg_pts=[(216, 346), (405, 263)],
        view_img=False,
        draw_tracks=True,
        line_thickness=1,
    )

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(output_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))
    
    while cap.isOpened():
        success, im0 = cap.read()
        if not success or im0 is None:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Track objects in the current frame
        tracks = model.track(im0, persist=True, show=False)
        
        if tracks:  # Ensure tracks are not empty or None
            nb_counter_right.im0 = im0
            nb_counter_left.im0 = im0
            nb_counter_thru.im0 = im0
            nb_counter_incomming.im0 = im0
            
            nb_counter_right.extract_and_process_tracks(tracks)
            nb_counter_left.extract_and_process_tracks(tracks)
            nb_counter_thru.extract_and_process_tracks(tracks)
            nb_counter_incomming.extract_and_process_tracks(tracks)

            print(nb_counter_right.class_wise_count)
            print(nb_counter_left.class_wise_count)
            print(nb_counter_thru.class_wise_count)
            print(nb_counter_incomming.class_wise_count)

            rect_height = h // 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 2
            color = (255, 255, 255)
            boxColors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (106, 90, 205)
            ]
            boxPositions = [
                ((0, 0), (w // 4, rect_height)),
                ((w // 4, 0), (w // 2, rect_height)),
                ((w // 2, 0), (3 * w // 4, rect_height)),
                ((3 * w // 4, 0), (w, rect_height))
            ]
            textPositions = [
                (10, rect_height // 2 + 5),
                (w // 4 + 10, rect_height // 2 + 5),
                (w // 2 + 10, rect_height // 2 + 5),
                (3 * w // 4 + 10, rect_height // 2 + 5)
            ]

            for index, key in enumerate(['PV', 'TTST', 'Duals', 'Twins']):
                rightData = nb_counter_right.class_wise_count.get(key, {})
                leftData = nb_counter_left.class_wise_count.get(key, {})
                thruData = nb_counter_thru.class_wise_count.get(key, {})
                incommingData = nb_counter_incomming.class_wise_count.get(key, {})

                combined_right = rightData.get('IN', 0) + rightData.get('OUT', 0)
                combined_left = leftData.get('IN', 0) + leftData.get('OUT', 0)
                combined_thru = thruData.get('IN', 0) + thruData.get('OUT', 0)
                combined_incoming = incommingData.get('IN', 0) + incommingData.get('OUT', 0)
                
                cv2.rectangle(im0, boxPositions[index][0], boxPositions[index][1], boxColors[index], -1)
                text = f"{key} : LEFT = {combined_left} -- RIGHT = {combined_right} -- THRU = {combined_thru} -- INCOMING = {combined_incoming}"
                cv2.putText(im0, text, textPositions[index], font, font_scale, color, font_thickness)
                
                # cv2.rectangle(im0, (w // 4, 0), (w // 2, rect_height), (0, 255, 0), -1)     # Second box
                # cv2.rectangle(im0, (w // 2, 0), (3 * w // 4, rect_height), (0, 0, 255), -1) # Third box
                # cv2.rectangle(im0, (3 * w // 4, 0), (w, rect_height), (255, 255, 0), -1)    # Fourth box
            
            nb_counter_left.display_frames()
            video_writer.write(im0)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            print(video_path, 'Video Path')
            print('Total Frames', total_frames, 'Current Frame', current_frame)
            
            # Optional: Limit the number of frames for testing purposes
            # if cap.get(cv2.CAP_PROP_POS_FRAMES) >= 200:
            #     print('Frames reached limit')
                # break
    
    vehicle_short_term_data = [
        nb_counter_right.class_wise_count,
        nb_counter_left.class_wise_count,
        nb_counter_thru.class_wise_count,
        nb_counter_incomming.class_wise_count
    ]
    new_data = copy.deepcopy(vehicle_short_term_data)
    save_to_excel_with_timestamps(new_data)

    # Release video resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()



# Initialize a list to store counts
vehicle_data = {vehicleType: [] for vehicleType in model.names.values()}
start_time = time.time()

# Directory containing the videos
video_dir = "videos"
output_excel_path = 'traffic_count_results.xlsx'

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# process_video("116/1_20240408-235702_1030h.avi")

#Process each video in the directory
for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4') or video_file.endswith('.avi'):  # Adjust the file extension if needed
        print(video_file)
        video_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"output_{os.path.basename(video_file)}")
        print(f"Processed {video_path}")
        print(f"Output Path {output_path}")
        process_video(video_path, output_path)

# Save count data to an Excel file
# total_objects_counted = max(object_counts) if object_counts else 0
# count_data = pd.DataFrame({'Total Objects Counted': [total_objects_counted]})
# count_data.to_excel("object_counts.xlsx", index=False)
