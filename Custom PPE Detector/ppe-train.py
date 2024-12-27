
######################
# TRAINING YOLO MODEL (Need to add usage of GPU instead of CPU)
######################

from ultralytics import YOLO

# Just checks installations on google collab ipynb file
# !yolo task=detect mode=predict model=yolov8l.pt conf=0.25 source='https://ultralytics.com/images/bus.jpg'

# Custom data training
# imgsz is image size
# !yolo task=detect mode=train model=yolov8n.pt conf=0.25 source=../content/drive/MyDrive/Datasets/ConstructionSafety/data.yaml epochs=50 imgsz=640


from ultralytics import YOLO

# Path to your data.yaml file
data_yaml_path = r"C:\Users\ringk\Downloads\ConstructionSafety\data.yaml"

# Initialize YOLO model (you can specify a different model if needed)
model = YOLO('../Yolo-Weights/yolov8n.pt')

# Train the model
model.train(
    data=data_yaml_path,  # path to the data.yaml file
    epochs=1,             # number of epochs to train
    imgsz=640,            # image size
    project='../runs', # where to save the training results
    name='exp'            # name of the experiment
)
