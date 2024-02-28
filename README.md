# AITrafficCounters - AI services
Installation:
Installing on the host machine

Step 1: Install Ultralytics.

Step 2: Clone the repository.

!git clone https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking.git


Step 3: Install the required packages using the command below.

Step 4: Set the path to install DeepSORT.

cd /YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect

Step 5: Download DeepSORT for tracking detected vehicles.

!gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"

Step 6: Unzip the downloaded file.

!unzip 'deep_sort_pytorch.zip'


Step 7: Run the script using the command below. Before doing so, replace the "predict.py" file with the updated "predict.py" 
and change the video path. If you want to see live detection, set "show" to True.

!python predict.py model=yolov8l.pt source="resized_video.mp4" show=True # for pretrained model


Make sure that your model and script must be in same folder
For custom trained model
!python predict.py model=best.pt source="resized_video.mp4" show=True

