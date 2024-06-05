import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#
def detect_circles(frame):
    """
    Circle detection using Hough circle transform

    Args: 
        frame (uint8): Input frame

    Returns:
        frame (uint8): Frame with detected circles annotated
        circles (int32): circles (list of tuples): List of detected circles with their positions and colors (x, y, color)
    """
    #convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blurred = cv.GaussianBlur(gray, (9, 9), 2)
    #detect circles
    circles=cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100)

    #store coordinates of circles
    detected_circles=[]
    #convert circle parameters x,y,r to integers
    if circles is not None:
        circles=np.round(circles[0,:]).astype("int")
        for (x, y, r) in circles:
            # Sample the color in the center of the circle
            color = frame[y, x].tolist()
             # Draw the circumference of the circle.
            labelColor=(0, 128, 255)
            cv.circle(frame, (x,y), r, labelColor, 4)
            #label center coordinates of the circle
            cv.circle(frame, (x,y), 5, (0, 0, 255), -1)
            #write label for detected circle
            font = cv.FONT_HERSHEY_SIMPLEX 
            origin=(x-50, y-110)
            cv.putText(frame, f'Circle ({color})', origin, font, 0.5, color, 1)
            #append coordinates to list
            detected_circles.append((x,y, color))

    return frame, detected_circles

#detect rectangles
def detect_rectangles(frame):
    """
    Detects rectangles using contour detection.

    Arg:
        frame (uint8): img frame input
    
    Returns:
        frame (uint8)
        rectangles(int32)
    """
    #convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Blur to reduce noise
    blurred = cv.GaussianBlur(gray, (9, 9), 2)
    #detect edges
    edges=cv.Canny(blurred, 50, 150)
    #find contours
    contours, _=cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    rectangles=[]
    for cnt in contours:
        #max distance from contour to approximated contour
        epsilon = 0.02 * cv.arcLength(cnt, True)
        #approximates contour to a polygon
        approx=cv.approxPolyDP(cnt, epsilon,True)
        #check if polygon has 4 vertices
        if len(approx)==4:
            #calculates min up-right bounding rectangle for the point set
           x,y,w,h=cv.boundingRect(approx)
           #calculate center point
           centerX=int(x+w/2)
           centerY=int(y+h/2) 
           textX=int(x-w/2)
           textY=int(y-h/2)
           labelColor=(0, 128, 255)
           #Sample the color in the center of the rectangle
           color = frame[centerY, centerX].tolist()
           cv.rectangle(frame, (x,y), (x+w,y+h), labelColor, 2)
           font = cv.FONT_HERSHEY_SIMPLEX
           cv.putText(frame, f'Rectangle ({color})', (textX,textY), font, 0.5, color, 1)
           #append center coordinates to list
           rectangles.append((centerX, centerY, color))
        
    return frame, rectangles


#track objects
def track_objects(frame,positions):
    """
    Track objects across frames from stored coordinates

    Arg:
        positions(dict): history of individual objects' positions based on the color

    Returns:
        frame(uint8): frame with tracked positions
    """

    # Loop through the dictionary and store values in a list for each key
    for key in positions.keys():
        prev_value=positions[key][0]
        for value in positions[key]:
            #convert int color to tuple RGB
            r = (key >> 16) & 0xFF
            g = (key >> 8) & 0xFF
            b = key & 0xFF
            color=(b,g,r)
            #check if the new position point is none -> missing object in the current frame
            if value == None and prev_value != None:
                #handle missing object
                x,y= prev_value
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(frame, 'Missing object', (x,y), font, 0.5, color, 1)
                continue
            #Draw a filled circle to represent a point
            x,y = value
            cv.circle(frame, (x,y), 5, color, thickness=2)
            #draw line between points
            cv.line(frame, prev_value, value, color, thickness=2 )
            prev_value=value

    return frame

#function to process output of the detection
def process_output(new_objects, positions):
    """
        Samples point coordinates from each frame and stores them based on the color.
        Args:
            new_objects(list): circles or rectangles
            positions(dict): previous positions
        Returns:
            positions(dict): dictionary of objects with keys colors and corresponding point coordinates

    """
    #separate objects based on color and store in the dictionary
    for circle in (new_objects):
        x, y, color = circle
        color=tuple(color)
        # Convert color tuple to integer format (RGB)
        color_int = (color[2] << 16) + (color[1] << 8) + color[0]
        #print(f"Color tuple: {color}")
        #print(f"Color integer (RGB): {color_int}")
        #check if same object already detected
        if color_int not in positions.keys():
            positions[color_int]=[]
        #store new coordinates for each object separately
        positions[color_int].append((x,y))
    # Print the dictionary
    for index, obj_positions in positions.items():
        print(f"Color {index}: {obj_positions}")

    return positions

# path visualization tool in plot
def path_visualization(positions):
    plt.figure()
    # Loop through the dictionary and store values in a list for each key
    for key in positions.keys():
        prev_value=positions[key][0]
        for value in positions[key]:
            #convert int color to tuple RGB
            r = (key >> 16) & 0xFF
            g = (key >> 8) & 0xFF
            b = key & 0xFF
            color=(r/255,g/255,b/255)
            #check if the new position point is none -> missing object in the current frame
            if value == None and prev_value != None:
                #handle missing object
                x,y= prev_value
                plt.plot(x,y, 'X', color='r')
                continue
            #Draw a filled circle to represent a point
            x,y = value
            plt.plot(x,y,'o',color=color)
            
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of paths')

    # Show the plot
    plt.grid(True)
    plt.show()

    return