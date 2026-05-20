import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math 
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn


import cnn
from cnn import CNNModel, class_mapping, val_loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#mean and std of this Monkey Species dataset
mean = [0.4368, 0.4336, 0.3294]  
std = [0.2457, 0.2413, 0.2447]

model = CNNModel()

# Load the best model weights
model.load_state_dict(torch.load("best.pt"))
model.to(DEVICE)
#inference
def prediction(model, val_loader):

    model.eval()

    all_images, all_labels = [], []
    all_pred_indices, all_pred_probs = [], []

    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        with torch.inference_mode():
            outputs = model(images)

        prob = F.softmax(outputs,dim=1)
        pred_indices = prob.max(dim=1)[1]
        pred_probs = prob.max(dim=1)[0]

        all_images.append(images.cpu())
        all_labels.append(labels.cpu())
        all_pred_indices.append(pred_indices.cpu())
        all_pred_probs.append(pred_probs.cpu())

    return (torch.cat(all_images).numpy(),
            torch.cat(all_labels).numpy(),
            torch.cat(all_pred_indices).numpy(),
            torch.cat(all_pred_probs).numpy())

def denormalize(image):
    mean_ar = np.array(mean)
    std_ar = np.array(std)
    image = image * std_ar + mean_ar
    return np.clip(image, 0,1)

def visualise_predictions(
    sample_images,
    sample_gt_labels,
    pred_indices,
    pred_probs,
    class_mapping,
    num_images=5
):

    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))

    # pick unique random indices (avoid duplicates)
    indices = random.sample(range(len(sample_images)), num_images)

    for i, idx in enumerate(indices):

        # image format: (C,H,W) → (H,W,C)
        image = sample_images[idx].transpose(1, 2, 0)

        label = sample_gt_labels[idx]
        pred_idx = pred_indices[idx]
        pred_prob = pred_probs[idx]

        # denormalize
        image = denormalize(image)

        # correctness check
        correct = label == pred_idx
        mark = "✔" if correct else "✘"

        axes[i].imshow(image)
        axes[i].axis('off')

        gt_name = class_mapping[label]
        pred_name = class_mapping[pred_idx]

        axes[i].set_title(
            f"{mark} GT: {gt_name}\nPred: {pred_name} ({pred_prob:.2f})",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()

val_images, val_gt_labels, pred_indices, pred_probs = prediction(model, val_loader)
visualise_predictions(val_images, val_gt_labels, pred_indices, pred_probs,class_mapping, num_images = 5)

cm = confusion_matrix(y_true=val_gt_labels, y_pred = pred_indices)

plt.figure(figsize= [10,5])
sn.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=class_mapping.values(),
    yticklabels=class_mapping.values()
)
plt.xlabel("Predicted")
plt.ylabel("Targets")
plt.title(f"Confusion Matrix", color="gray")
plt.show()