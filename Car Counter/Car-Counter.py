from ultralytics import YOLO 
import cv2 
import cvzone #Displays detections, not required
import math
import os


model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = model.names #Dict to list

# cap = cv2.VideoCapture(0) #0 is the webcam, 1 is for multiple webcams
# cap.set(3, 640) #Width, which is prop #3 Could also be 640 by 480, or 1280 by 720, just pick one
# cap.set(4, 480) #Height

current_dir = os.getcwd() # Get the current working directory
cap = cv2.VideoCapture(fr'{current_dir}\Project1\Videos\cars.mp4')



while True:
    success, img = cap.read()
    if not success: #If cap is undefined, then we just break it immediately so we do not crash
        break
    results = model(img, stream=True) #Stream=True uses generators, which is faster. It is recommended to set this to true.
    
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
            cvzone.cornerRect(img, (x1, y1, w, h), l=5) #Fancy rectangle, bounding box

            # Confidence
            conf = box.conf[0]
            conf = math.ceil((box.conf[0]*100))/100 #2 decimal places rounding
        
            # ClassName
            cls = box.cls[0] #The ID of the class
            # max(0,x1) means it will not go above 0, so it will stay within the image
            # scale makes the text smaller
            # default thickness is 3, we make it smaller so we can still read
            cvzone.putTextRect(img, f'{classNames[int(cls)]} {conf}', (max(0,x1), max(35,y1)), scale=0.6, thickness=1, offset=3)
            
            
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #If q is pressed, break the loop
        break

    # cv2.waitKey(1) #1ms delay

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


