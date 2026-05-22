from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def prediction(model_type, img_path, display_result = False, task : str = None):

    model = YOLO(model_type)
    result = model(img_path, save = True, conf=0.5)

    for r in result:
        pred_img_path = f"{r.save_dir}/{img_path}"
        pred = cv2.cvtColor(cv2.imread(pred_img_path), cv2.COLOR_BGR2RGB)
        plt.imshow(pred)
        plt.axis('off')
        plt.title(f"YOLO11 - {task}")
    plt.show()

    if display_result:
       print(result)

# Load the pre-trained YOLOv11-Medium model
img_path = "football.jpg"
model_type = "yolo11l.pt"

prediction(model_type, img_path, task = "Object Detection")


img_path = "zebra.jpg"
prediction(model_type, img_path, display_result= True, task = "Object Detection")

img_path = "monitor.jpg"
prediction(model_type, img_path, task = "Object Detection")

img_path = "kitchen.jpg"
prediction(model_type, img_path, task = "Object Detection")

img_path = "cat-dog.jpg"
prediction(model_type, img_path, task = "Object Detection")

model_type = "yolo11l-pose.pt"

img_path = "football.jpg"

# Predict with the model
prediction(model_type, img_path, task = "Pose Estimation")

# Load a model
model_type = "yolo11l-seg.pt"

img_path = "football.jpg"

# Predict with the model
prediction(model_type, img_path, task = "Instance Segmentation", display_result = False)

model_type = "yolo11l-cls.pt"

img_path = "tiger.jpg"

# Predict with the model
prediction(model_type, img_path, task = "Classification")

model_type = "yolo11l-obb.pt"
img_path = "boat.jpg"

prediction(model_type, img_path, task = "OBB")