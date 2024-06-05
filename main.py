import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
#
from functions import detect_rectangles
from functions import detect_circles
from functions import process_output
from functions import track_objects
from functions import path_visualization


def main():
    #initialize argument parser
    ap = argparse.ArgumentParser(description="load a video file")
    # Add an argument for the video file path
    ap.add_argument("video_path", help="path to the video file")
    args = ap.parse_args()
    cap = cv.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.video_path}")
        return
    #cap = cv.VideoCapture('luxonis_task_video.mp4')
    #visualization output
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    #generate video file
    output = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
    #initialize variables
    circle_positions={}
    previous_circles={}
    rectangles_positions={}
    previous_rectangles={}
    ##keep looping, while video s running
    while cap.isOpened():
        ret, frame=cap.read()
        #if we are viewing a video and we did not grab a frame,
	    # then we have reached the end of the video
        if ret == False:
            break
        #detect and label objects
        processed_frame, circles=detect_circles(frame)
        processed_frame, rectangles=detect_rectangles(frame)
        #display frame with detected objects
        #cv.imshow('Frame', processed_frame)
        #process detected objects
        circle_positions=process_output(circles, previous_circles)
        rectangles_positions=process_output(rectangles, previous_rectangles)
        #update list of detected positions of circles
        previous_circles=circle_positions
        #update list of detected positions of circles
        previous_rectangles=rectangles_positions

        #track circles' paths
        frame=track_objects(processed_frame, circle_positions)
        #track rectangles' paths
        frame=track_objects(processed_frame, rectangles_positions)
        #display detection and tracking
        cv.imshow('Frame', frame)
        #write into video file
        output.write(frame)
       
        # Exit if 'Esc' key is pressed
        key = cv.waitKey(30) & 0xff
        if key == 27:  
            break
    
    cap.release()
    output.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
