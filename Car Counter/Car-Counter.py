from ultralytics import YOLO 
import cv2 
import cvzone #Displays detections, not required
import math
import os, time
from sort import *

model = YOLO('../Yolo-Weights/yolov8n.pt')

mask = cv2.imread(r'C:\Users\ringk\OneDrive\Documents\Object-Detection-Course\Info\Images\CarMask.png')

#Tracking
#max_age (Ex: If ID#1 is lost, how many frames do we wait to detect it back)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) 
# iou_threshold is the intersection over union, which is the overlap between the bounding boxes

#When it crosses a line, we make it count the obj

#Point1, point2
limits = [0, 200, 673, 200]
totalCount = []

classNames = model.names #Dict to list

# cap = cv2.VideoCapture(0) #0 is the webcam, 1 is for multiple webcams
# cap.set(3, 640) #Width, which is prop #3 Could also be 640 by 480, or 1280 by 720, just pick one
# cap.set(4, 480) #Height

def vidDimensions(vidPath):
    # Path to your video file
    video_path = vidPath

    # Open the video file
    capVid = cv2.VideoCapture(video_path)

    # Get the width and height of the video
    width = int(capVid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capVid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video dimensions: {width} x {height}")

    # Release the video capture object
    capVid.release()


current_dir = os.getcwd() # Get the current working directory
carVideo = r'C:\Users\ringk\OneDrive\Documents\Object-Detection-Course\Info\Videos\cars.mp4'
cap = cv2.VideoCapture(carVideo)
vidDimensions(carVideo)

while True:
    success, img = cap.read()
    if not success: #If cap is undefined, then we just break it immediately so we do not crash
        print("Failed to read from the video")
        break
    imgRegion = cv2.bitwise_and(img, mask) #Masking the image
    
    # Importing within the while loop so the graphics dont get bad
    imgGraphics = cv2.imread(r'C:\Users\ringk\OneDrive\Documents\Object-Detection-Course\Car Counter\graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics) #Overlaying the graphics on the image
    
    
    results = model(imgRegion, stream=True) #Stream=True uses generators, which is faster. It is recommended to set this to true.
    detections = np.empty((0,5))
    #Bounding boxes 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # OPENCV --> Bounding Box
            x1, y1, x2, y2 = box.xyxy[0] #Easier to input into opencv
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) # (x1, y1) = top left, (x2, y2) = bottom right, (255, 0, 255) = color, 3 = thickness

            # CVZONE --> Bounding Box
            w, h = x2-x1, y2-y1
            # When it's smaller it's not putting the rectangle properly, so we specify the length

            # Confidence
            conf = box.conf[0]
            conf = math.ceil((box.conf[0]*100))/100 #2 decimal places rounding
        
            # ClassName
            cls = box.cls[0] #The ID of the class
            currentClass = classNames[int(cls)]
            # max(0,x1) means it will not go above 0, so it will stay within the image
            # scale makes the text smaller
            # default thickness is 3, we make it smaller so we can still read
            if (currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus' or currentClass == 'motorbike') and conf >= 0.3: #Only the class is has a confidence and a class
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0,x1), max(35,y1)), scale=0.6, thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=5) #Fancy rectangle, bounding box
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack([detections, currentArray]) #Append (vstack) array to detections

            # We see how objects are classified the best in the middle, and we need to ignore some areas
    
    # [x1, y1, x2, y2, score]
    resultsTracker = tracker.update(detections)    #Update it with list of detections
    
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=5, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{currentClass} {id}', (max(0,x1), max(35,y1)), scale=2, thickness=3, offset=10)
        w, h = x2-x1, y2-y1

        # Center
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED) #Just ensuring accuracy
        
        # Check lim of x and y. If the limit is straight forward
        # But cars travel fast, so we'd have a box region to check it
        # Balance the region, but also do not re-count IDs
        # The -+ signs offset makes it detect it early (higher value) or late (lower value)
        if (limits[0] < cx < limits[2]) and (limits[1] - 15 < cy < limits[1] + 15):
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)

    
    # cvzone.putTextRect(img, f'Count: {len(totalCount)}', (100, 50))
    cv2.putText(img, str(len(totalCount)), (100, 70), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    # cv2.waitKey(0) #Everytime we press the keyboard it will proceed

    if cv2.waitKey(1) & 0xFF == ord('q'): #If q is pressed, break the loop
        break
    # time.sleep(100)
    # cv2.waitKey(1) #1ms delay

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


