# AITrafficCounters

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3.3-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

AITrafficCounters is a traffic counting and classification system that utilizes YOLOv8 for vehicle detection and classification. The system processes videos from a specified directory, performs inference on each frame, annotates the frames with vehicle counts and classifications, and saves the annotated videos to an output directory. The system also saves the counts to an Excel file.

<p align="center">
  <img src="https://www.ultralytics.com/images/yolov8-logo.png" alt="YOLOv8" width="300">
</p>

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Script Overview](#script-overview)
  - [process_data Function](#process_data-function)
  - [save_to_excel_with_timestamps Function](#save_to_excel_with_timestamps-function)
  - [process_video Function](#process_video-function)
  - [Object Counters Initialization](#object-counters-initialization)
  - [Displaying Counts on Frames](#displaying-counts-on-frames)
- [Example Output](#example-output)
- [Customization](#customization)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Prerequisites

- Python 3.6+
- OpenCV
- Ultralytics YOLOv8
- Pandas

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Dan-445/AITrafficCounters.git
    cd AITrafficCounters
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Place the videos you want to process in the `videos` directory.** The system supports `.mp4` and `.avi` file formats by default. You can adjust the supported file extensions in the script if needed.

2. **Run the script:**
    ```bash
    python process_videos.py
    ```

3. **The annotated videos will be saved in the `output` directory.** The vehicle counts will be saved in an Excel file named `traffic_count_results.xlsx`.

## Script Overview

### `process_videos.py`

This script performs the following steps:

1. **Initializes the YOLOv8 model** using a pre-trained weights file `countbest.pt`.
2. **Processes each video in the `videos` directory:**
    - Opens the video file.
    - Initializes the object counters for different vehicle directions (incoming, left, thru, right).
    - Reads each frame, performs YOLO inference, and annotates the frame with vehicle counts.
    - Writes the annotated frame to the output video.
3. **Saves the vehicle counts to an Excel file.**

### `process_data` Function

This function processes the data from the object counters and appends the combined values to a global structure for later saving to an Excel file.

### `save_to_excel_with_timestamps` Function

This function converts the global structure of vehicle counts to a pandas DataFrame and saves it to an Excel file.

### `process_video` Function

This function processes a single video file, performs YOLO inference on each frame, annotates the frame, and saves the annotated frames to an output video file.

### Object Counters Initialization

The object counters are initialized for different vehicle directions (incoming, left, thru, right) with specified regions of interest.

### Displaying Counts on Frames

The script annotates each frame with the vehicle counts for different classes (PV, TTST, Duals, Twins) in different directions (left, right, thru, incoming). The counts are displayed in colored rectangles at the top of the frame.

## Example Output

Annotated frames will look something like this:

PV : LEFT = 10 -- RIGHT = 12 -- THRU = 15 -- INCOMING = 8
TTST : LEFT = 3 -- RIGHT = 5 -- THRU = 2 -- INCOMING = 1
Duals : LEFT = 7 -- RIGHT = 8 -- THRU = 9 -- INCOMING = 4
Twins : LEFT = 2 -- RIGHT = 3 -- THRU = 5 -- INCOMING = 0


## Customization

You can customize the following parts of the script:
- **Regions of Interest:** Adjust the coordinates in the `reg_pts` parameter for each object counter.
- **Supported File Extensions:** Modify the file extensions in the `if` statement to support additional video formats.
- **Output File Paths:** Change the `output_dir` and `output_excel_path` variables to customize the locations of the output files.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO model and training framework.
- [OpenCV](https://opencv.org/) for video processing and frame annotation.
- [Pandas](https://pandas.pydata.org/) for data manipulation and saving to Excel.

<p align="center">
  <img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_no_text.png" alt="OpenCV" width="150">
  <img src="https://pandas.pydata.org/static/img/pandas_secondary_white.svg" alt="Pandas" width="150">
</p>
