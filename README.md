# Real-Time Object Tracking Simulation Script

## Description

This script detects and tracks objects from a video file using OpenCV library. The video file path has to be provided as a command line argument.

###main.py
Loads video file and runs detection and tracking algorithm.
Outputs video file for object's path visualization.

###functions.py
Functions for detecting circles and rectangles, for processing detected objects and tracking.


## Usage

To run the script, use the following command:

```sh
python main.py path/to/your/video.mp4