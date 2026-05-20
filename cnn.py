import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.ops import Conv2dNormActivation

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time
import numpy as np
import random
import warnings
import os
from tqdm import tqdm
import pandas as pd


#Set seed for reprod
# ucibilty
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = True

set_seed(42)

@dataclass(frozen=True)
class TrainingConfig:
      ''' Configuration for Training '''
      batch_size: int = 32
      num_epochs: int = 100
      learning_rate: float = 1e-4

      log_interval: int = 1
      test_interval: int = 1
      data_root: str = "./"
      num_workers: int = 5
      device: str = "cuda"

train_config = TrainingConfig()
DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"

train_root = os.path.join("10_Monkey_Species", "training", "training")
val_root = os.path.join(train_config.data_root, "10_Monkey_Species", "validation", "validation")

df = pd.read_csv(os.path.join("10_Monkey_Species", "monkey_labels.txt"))
#pd.set_option('display.max_columns', None)
#print(df)  

#mean and std of this Monkey Species dataset
mean = [0.4368, 0.4336, 0.3294]  
std = [0.2457, 0.2413, 0.2447]

img_size = (224,224)

def get_transforms(img_size, mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)

    train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(45, (0.1,0.3), (0.5,0.75)),
        transforms.RandomErasing(p=0.4),
        transforms.ToTensor(),
        normalize 
    ])

    val = transforms.Compose([

        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize
    ])

    return train, val

train_transforms, val_transforms = get_transforms(img_size, mean, std)

train_data = datasets.ImageFolder(root=train_root, transform=train_transforms)
val_data = datasets.ImageFolder(root=val_root, transform=val_transforms)

train_loader = DataLoader(
    train_data,
    shuffle=True,
    batch_size=train_config.batch_size,
    num_workers=train_config.num_workers
)

val_loader = DataLoader(
    val_data,
    shuffle= False,
    batch_size= train_config.batch_size,
    num_workers=train_config.num_workers
)

class_mapping = {

    0: "mantled_howler",
    1: "patas_monkey",
    2: "bald_uakari",
    3: "japanese_macaque",
    4: "pygmy_marmoset",
    5: "white_headed_capuchin",
    6: "silvery_marmoset",
    7: "common_squirrel_monkey",
    8: "black_headed_night_monkey",
    9: "nilgiri_langur"
}

def visualize_images(dataloader, num_images = 20):
    fig = plt.figure(figsize=(10,10))

    #Iterate over the first batch
    images, labels = next(iter(dataloader))
    # print(images.shape)

    num_rows = 4
    num_cols = int(np.ceil((num_images / num_rows)))

    for idx in range(min(num_images, len(images))):
        image, label = images[idx], labels[idx]


        ax = fig.add_subplot(num_rows, num_cols, idx+1, xticks = [], yticks = [])

        image = image.permute(1,2,0)

        #Normalize the image to [0,1] to display

        image = (image - image.min()) / (image.max() - image.min())
        ax.imshow(image, cmap="gray")  # remove the batch dimension
        ax.set_title(f"{label.item()}: {class_mapping[label.item()]}")

    fig.tight_layout()
    plt.show()

visualize_images(train_loader, num_images = 16)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._model = nn.Sequential(

            # ---------------- Block 1 ----------------
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2), 

            # ---------------- Block 2 ----------------
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # ---------------- Block 3 ----------------
            Conv2dNormActivation(128, 256, kernel_size=3),
            Conv2dNormActivation(256, 256, kernel_size=3),
            nn.MaxPool2d(2),

            Conv2dNormActivation(256, 512, kernel_size=3),
            nn.MaxPool2d(2),

            # ---------------- Head ----------------
            nn.AdaptiveAvgPool2d(output_size=(3,3)),
            nn.Flatten(),

            nn.Linear(in_features = 512*3*3, out_features = 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(in_features = 256, out_features = 10)
        )

    def forward(self, x):
        return self._model(x)
    
model = CNNModel()


#train function
def train(model, train_loader, optimizer):
    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), 100 * correct / total

#validation function
def validation(model, val_loader):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(val_loader), 100 * correct / total


def main(model, train_loader, val_loader, optimizer):

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    model.to(DEVICE)

    for epoch in range(train_config.num_epochs):

        train_loss, train_acc = train(model, train_loader, optimizer)
        val_loss, val_acc = validation(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{train_config.num_epochs} "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best.pt")
            print("Saving best model 💾")

    return train_losses, train_accs, val_losses, val_accs

if __name__ == "__main__":

    model = CNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)

    train_losses, train_accs, val_losses, val_accs = main(
        model,
        train_loader,
        val_loader,
        optimizer,
    )