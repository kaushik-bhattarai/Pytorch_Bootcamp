import torch 
import torchvision
from torchvision import models
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

import os
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

from downloader import download_and_unzip

URL = r"https://www.dropbox.com/s/8srx6xdjt9me3do/TF-Keras-Bootcamp-NB07-assets.zip?dl=1"
folder_path = "/home/kaushik/pytorch_bootcamp/assets"
asset_zip_path = os.path.join(folder_path, "PyTorch""-Bootcamp-NB07-assets.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

#print(dir(models))

# Download imagenet classes text file.
#!wget -q  'https://raw.githubusercontent.com/Lasagne/Recipes/master/examples/resnet50/imagenet_classes.txt' -O'imagenet_classes.txt'

#preprocessing the inputs
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#load resnet50 model
model = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

#put model in eval mode for inference
model.eval()
print(model)

# Read the image.
img = Image.open("assets/images/baseball-player.png")
plt.imshow(img)
plt.axis('off')
plt.show()

img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0) #Add batch dimension [C,H,W] --> [B,C,H,W]

# Carry out inference
out = model(batch_t)
print(out.shape) # [B,num_classes]

def visualize_predictions(img, class_name, conf):
    """
    Function to visualize results:
    :param img: PIL Image
    :param class_name: Class name string
    :param conf: Prediction confidence string
    """
    bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_h, img_w = bgr_img.shape[:2]

    # Define font scale and thickness based on image height
    font_scale = max(0.003 * img_h, 0.5)
    thickness = max(1, int(img_h / 200))

    text = f"{class_name}, {conf}%"

    # Calculate text size to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (img_w - text_w) // 2
    text_y = (img_h + text_h) // 10

    cv2.putText(
        img=bgr_img,
        org=(text_x, text_y),
        text=text,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        color=(0, 0, 255),
        fontScale=font_scale,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.show()

    # Load labels
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]


class_name = classes[indices[0][0]]
conf = f"{percentage[indices[0][0]].item():.1f}"
visualize_predictions(img, class_name, conf)

def prediction(img_path, model):
    model.eval()
    img = Image.open(img_path)
    img_t = transform(img).unsqueeze(0)
    out = model(img_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    class_name = classes[indices[0][0]].split(',')[0]
    conf = f"{percentage[indices[0][0]].item():.1f}"

    return img, class_name, conf

for img_path in os.listdir("assets/images"):
    img_path = os.path.join("assets/images", img_path)
    img, class_name, conf = prediction(img_path, model)

    visualize_predictions(img, class_name, conf)