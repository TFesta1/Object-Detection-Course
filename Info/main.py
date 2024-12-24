# pip install cvzone==1.5.6 ultralytics==8.0.26 hydra-core>=1.2.0 matplotlib>=3.2.2 numpy>=1.18.5 opencv-python==4.5.4.60 Pillow>=7.1.2 PyYAML>=5.3.1 requests>=2.23.0 scipy==1.4.1 torch==1.7.0 torchvision>=0.8.1 tqdm==4.64.0 filterpy==1.4.5 scikit-image==0.19.3 lap==0.4.0
# It SHOULD be the above versions, but the pip install does not work for this
# pip install cvzone ultralytics hydra-core matplotlib numpy opencv-python Pillow PyYAML requests scipy torch torchvision tqdm filterpy scikit-image lapx
# Instead of pip install lap, using pip install lapx since it has the same functionality as lap


import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


model = YOLO('yolov8n.pt') #Weights (nano, medium, large, the n here stands for nano) just type in name and it would download the weights
#https://docs.ultralytics.com/models/yolov8/#models --> has the table with models and what it does

"""
Options: n = nano, s = small, m = medium, v = large, x = xlarge

1. yolov8n = Detection = Rectangles around objects
2. yolov8-seg = Segmentation = Actually draws the objects as they are (not just rectangles)
3. yolov8-pose = Pose/Keypoints = Keypoints (joints, eyes, nose, etc) to estimate a person's pose
4. yolov8-obb = Oriented Detection = Rectangles around objects rotated to fit their orientation
5. yolov8-cls = Classification = Labels an entire image with a single class (not object detection) --> cat, dog, etc
"""

# results = model(r"C:\Users\ringk\OneDrive\Documents\Object-Detection-Course\Info\Images\1.png", show=True) #Show=True see the image
# cv2.waitKey(0) #0 means unless the user inputs, do not do anything