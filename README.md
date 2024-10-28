

# AITrafficInsights

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3.3-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**AITrafficInsights** is an advanced AI-powered traffic counting and classification system that uses YOLOv8 for precise vehicle detection, classification, and tracking. The system processes video footage, identifies vehicles, counts them based on direction and type, and stores the results in an Excel file for easy analysis. This tool is ideal for traffic analysts, researchers, and urban planners needing detailed vehicle data insights.

<p align="center">
  <img src="https://www.ultralytics.com/images/yolov8-logo.png" alt="YOLOv8" width="300">
</p>

## Table of Contents
- [Key Features](#key-features)
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

## Key Features

- **Accurate Vehicle Detection:** Leverages YOLOv8 for highly accurate, real-time detection of various vehicle types.
- **Directional Counting:** Classifies and counts vehicles based on their direction (e.g., left, right, thru, incoming).
- **Detailed Annotations:** Annotates each frame with vehicle count and classification, providing a clear visual representation of traffic flow.
- **Automated Excel Reports:** Compiles vehicle counts into an organized Excel file with timestamps for further analysis.
- **Flexible Customization:** Easily adaptable for different regions, supported file types, and output locations.

## Prerequisites

- Python 3.6 or higher
- OpenCV
- Ultralytics YOLOv8
- Pandas

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Dan-445/AITrafficInsights.git
    cd AITrafficInsights
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Place the videos you want to process in the `videos` directory.**  
   - The system supports `.mp4` and `.avi` file formats by default. Modify the script to support additional formats if necessary.

2. **Run the script:**
    ```bash
    python process_videos.py
    ```

3. **Output:**  
   - Annotated videos with vehicle counts are saved in the `output` directory.
   - Vehicle counts, along with timestamps, are saved in an Excel file named `traffic_count_results.xlsx` for further analysis.

## Script Overview

### `process_videos.py`

This script manages the main flow of the AITrafficInsights system, performing the following steps:

1. **Model Initialization**:  
   - Loads the YOLOv8 model with pre-trained weights (`countbest.pt`) for detecting various vehicle types.
2. **Video Processing**:  
   - Iterates over videos in the `videos` directory.
   - Applies YOLOv8 detection on each frame to classify and count vehicles by type and direction.
   - Annotates frames with vehicle counts.
3. **Result Storage**:  
   - Saves each annotated frame in an output video.
   - Exports vehicle counts to an Excel file for a structured overview of traffic data.

### `process_data` Function

This function consolidates and appends vehicle count data to a global structure, preparing it for output to an Excel file.

### `save_to_excel_with_timestamps` Function

Generates a pandas DataFrame from vehicle count data and exports it to an Excel file with timestamps, providing a structured view of the traffic data.

### `process_video` Function

Handles the YOLO inference on individual video files, performing the following:
- Annotates frames based on vehicle detection results.
- Saves annotated frames in an output video file.

### Object Counters Initialization

Object counters are initialized for different vehicle directions (incoming, left, thru, right), each with specific regions of interest.

### Displaying Counts on Frames

The script annotates each frame with the vehicle counts for different classes (e.g., PV, TTST, Duals, Twins) based on direction. These counts are displayed in colored rectangles on the frame.

## Example Output

Annotated frames will display as follows:

PV : LEFT = 10 -- RIGHT = 12 -- THRU = 15 -- INCOMING = 8 TTST : LEFT = 3 -- RIGHT = 5 -- THRU = 2 -- INCOMING = 1 Duals : LEFT = 7 -- RIGHT = 8 -- THRU = 9 -- INCOMING = 4 Twins : LEFT = 2 -- RIGHT = 3 -- THRU = 5 -- INCOMING = 0


## Customization

Customize the following parts of the script as needed:
- **Regions of Interest**: Adjust the coordinates in the `reg_pts` parameter for each object counter.
- **Supported File Extensions**: Modify the file extensions in the `if` statement to support additional video formats.
- **Output File Paths**: Change the `output_dir` and `output_excel_path` variables to customize the locations of the output files.

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

