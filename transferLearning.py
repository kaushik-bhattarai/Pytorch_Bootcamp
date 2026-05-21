import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import time 
import numpy as np
import matplotlib.pyplot as plt
import os 

import pandas as pd
from PIL import Image
from torchvision import datasets, models, transforms
from torchinfo import summary
from torch.utils.data import DataLoader

from downloader import download_and_unzip

device = "cuda" if torch.cuda.is_available() else "cpu"

plt.style.use('ggplot')

# ── Paths & constants (always defined so inference can import them) ───────────
URL = r'https://www.dropbox.com/s/0ltu2bsja3sb2j4/caltech256_subset.zip?dl=1'
folder_path = "/home/kaushik/pytorch_bootcamp/assets"
asset_zip_path = os.path.join(folder_path, "caltech256_subset.zip")

dataset = 'assets/caltech256_subset'
train_dir = os.path.join(dataset, 'train')
valid_dir = os.path.join(dataset, 'valid')
test_dir  = os.path.join(dataset, 'test')
batch_size = 32

#Transforms (always defined so inference can import them)
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'general': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

#Dataset & loaders (always defined so inference can import them

# Download dataset ZIP if it doesn't exist yet
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

data = {
    'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['general']),
    'test':  datasets.ImageFolder(root=test_dir,  transform=image_transforms['general'])
}

idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size  = len(data['test'])

train_data_loader = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=False)
test_data_loader  = DataLoader(data['test'],  batch_size=batch_size, shuffle=False)

num_classes = len(os.listdir(valid_dir))

#Model builder function (call this wherever you need a fresh model)
def build_model(num_classes):
    resnet50 = models.resnet50(weights='DEFAULT')
    for param in resnet50.parameters():
        param.requires_grad = False
    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1)
    )
    return resnet50.to(device)

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

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

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
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

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
            torch.save(model.state_dict(), "best_model.pt")

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


# ── Only runs when this file is executed directly, NOT when imported ──────────
if __name__ == "__main__":
    print(f"Number of classes:            {num_classes}")
    print(f"Class mapping:                {idx_to_class}")
    print(f"Number of training samples:   {train_data_size}")
    print(f"Number of validation samples: {valid_data_size}")
    print(f"Number of test samples:       {test_data_size}")

    resnet50  = build_model(num_classes)
    loss_func = nn.NLLLoss()
    optimizer = optim.SGD(resnet50.parameters(), lr=0.01, momentum=0.9)

    num_epochs = 25
    trained_model, history, best_epoch = train_and_validate(resnet50, loss_func, optimizer, num_epochs)

    torch.save(history, dataset + '_history.pt')

    history = np.array(history)

    plt.figure(figsize=(10, 7))
    plt.plot(history[:, 0:2])
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.savefig('loss_curve.png')
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(history[:, 2:4])
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_curve.png')
    plt.show()