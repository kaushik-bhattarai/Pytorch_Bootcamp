import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torchvision.ops import Conv2dNormActivation
from torchvision.models import mobilenet_v3_small


from dataclasses import dataclass
import seaborn as sn

import matplotlib.pyplot as plt
import time 
import numpy as np
import random
import warnings
import os
from tqdm import tqdm
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)  # Set the seed for CUDA (single GPU)
       torch.cuda.manual_seed_all(seed)  # Set the seed for CUDA (multiple GPUs)
       torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN
       torch.backends.cudnn.benchmark = True  # Enable benchmark mode for CuDNN

set_seed(42)

@dataclass(frozen=True)
class TrainingConfig:

    '''configuration for training'''

    batch_size: int = 32
    num_epochs: int = 25
    learning_rate: float = 1e-4
    num_workers: int = 5

    log_interval: int = 1
    test_interval: int = 1
    data_root: int = "./"
    DEVICE: str = "cuda"

train_config = TrainingConfig()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = "10_Monkey_Species"
#training and validation data path
train_root = "/home/kaushik/pytorch_bootcamp/10_Monkey_Species/training/training"
val_root = "/home/kaushik/pytorch_bootcamp/10_Monkey_Species/validation/validation"

df = pd.read_csv("/home/kaushik/pytorch_bootcamp/10_Monkey_Species/monkey_labels.txt", sep=",", skipinitialspace=True)
# Clean column names
df.columns = [col.strip() for col in df.columns]

# Clean string columns
for col in ["Label", "Latin Name", "Common Name"]:
    df[col] = (
        df[col]
        .str.replace("\t", " ", regex=False)
        .str.strip()
    )
    
#print(df)

# Mean and standard deviation values computed from the Monkey Species dataset
mean = [0.4368, 0.4336, 0.3294]  # Mean
std = [0.2457, 0.2413, 0.2447]  # Standard deviation 

# define train and validation/test data transformation
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'general': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
}

train_data = datasets.ImageFolder(root= train_root, transform= image_transforms['train'])
val_data = datasets.ImageFolder(root= val_root, transform= image_transforms['general'])

train_data_size = len(train_data)
valid_data_size = len(val_data)


'''
print(train_data.classes)
print(train_data.class_to_idx)
print(len(train_data))
'''
#creating Dataloader
train_loader = DataLoader(
    train_data,
    shuffle=True,
    batch_size= train_config.batch_size,
    num_workers=train_config.num_workers
)

val_loader = DataLoader(
    val_data,
    shuffle=False,
    batch_size= train_config.batch_size,
    num_workers= train_config.num_workers
)

#Map class names to class ids.
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

def visualize_images(dataloader, num_images=20):
    """
    Function to visualize a batch of images from the DataLoader.

    Parameters:
    dataloader (DataLoader): The PyTorch DataLoader containing image data.
    num_images (int): Number of images to visualize.
    """
    fig = plt.figure(figsize=(10, 10))  # Create a figure with a specified size

    # Retrieve the first batch from the DataLoader
    images, labels = next(iter(dataloader))

    num_rows = 4  # Define number of rows in the grid
    num_cols = int(np.ceil((num_images / num_rows)))  # Compute number of columns

    for idx in range(min(num_images, len(images))):  # Loop through the images up to num_images
        image, label = images[idx], labels[idx]  # Extract image and label

        ax = fig.add_subplot(num_rows, num_cols, idx + 1, xticks=[], yticks=[])  # Create subplot

        image = image.permute(1, 2, 0)  # Reorder dimensions for visualization (C, H, W) → (H, W, C)

        # Normalize the image to [0,1] for display
        image = (image - image.min()) / (image.max() - image.min())
        ax.imshow(image, cmap="gray")  # Display the image
        ax.set_title(f"{label.item()}: {class_mapping[label.item()]}")  # Set title with label info

    fig.tight_layout()  # Adjust layout for better spacing
    plt.show()  # Display the figure

# Call the function to visualize a sample of images from the training dataset
#visualize_images(train_loader, num_images=16)


#fineTuning using mobilenet_v3_small
#Model builder function (call this wherever you need a fresh model)
def build_model():
    mobilenet_v3_model = mobilenet_v3_small(weights= "DEFAULT")
    num_feature_layers = len(mobilenet_v3_model.features)
    print(f"number of feature layers: {num_feature_layers}")

    num_classifier_layers = len(mobilenet_v3_model.classifier)
    print(f"Number of classifier layers: {num_classifier_layers}")

    #Set requires_grad to True for all model parameters to allow training
    # Freeze the earlier layers to retain pretrained low-level features
    for param in mobilenet_v3_model.features[:10].parameters(): 
        param.requires_grad = False

    print(mobilenet_v3_model.classifier[3])

    # Replace the final linear layer of the classifier to match the number of classes in the dataset
    mobilenet_v3_model.classifier[3] = nn.Linear(in_features = 1024, out_features = 10, bias = True)

    # Check the final classifier after modifications
    print(mobilenet_v3_model.classifier[3])

    return mobilenet_v3_model.to(DEVICE)


# Training function
def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    """
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)

    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
        best_epoch: Epoch number with best validation loss
    """

    start = time.time()
    history = []
    best_loss = 100000.0
    best_epoch = None

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc  = 0.0
        valid_loss = 0.0
        valid_acc  = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for j, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "best_modelFT.pt")

        avg_train_loss = train_loss / train_data_size
        avg_train_acc  = train_acc  / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc  = valid_acc  / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()

        print("Epoch : {:03d}, Training: Loss - {:.4f}, Accuracy - {:.4f}%, \n\t\t"
              "Validation : Loss - {:.4f}, Accuracy - {:.4f}%, Time: {:.4f}s".format(
              epoch, avg_train_loss, avg_train_acc * 100,
              avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start))

    return model, history, best_epoch

if __name__ == "__main__":
    
    mobilenet_model  = build_model()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mobilenet_model.parameters(), lr=0.01)

    num_epochs = 25
    trained_model, history, best_epoch = train_and_validate(mobilenet_model, loss_func, optimizer, num_epochs)

    torch.save(history, dataset + '_history.pt')

    history = np.array(history)

    plt.figure(figsize=(10, 7))
    plt.plot(history[:, 0:2])
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig('assets/loss_curveFT.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(history[:, 2:4])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.savefig('assets/accuracy_curveFT.png')
    plt.show()