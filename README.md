# AITrafficCounters - AI services
Installation
1. Installing on the host machine
Step1. Install ByteTrack.

git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
Step2. Install pycocotools.

pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
Step3. Others

pip3 install cython_bbox

step4. Install ultralytics
!pip install ultralytics

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

step5. Import BYTETracker and STrack and set a parameter.
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

step6.Install supervision 
!pip install supervision==0.1.0

from IPython import display
display.clear_output()

import supervision
print("supervision.__version__:", supervision.__version__)

step7.Import supervision libraries for vehicle color segmentation ,vehicle pointing, generating video frames after detection sink a frames to video
%%capture
from supervision.draw.color import ColorPalette
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook

step8.After Installing and import all the required libraries
Download the model.

MODEL = "yolov8x.pt"

from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

Step9.Then Run the vehicle detection and box annotation cell.

step10.Resize a frame for fit to our model.

step11.Run the main script to label all detection on frames and save output progress in CSV format.