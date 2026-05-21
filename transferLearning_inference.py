import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transferLearning import (
    build_model, num_classes,
    test_data_loader, test_data_size,
    image_transforms, idx_to_class
)

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn as nn
loss_func = nn.NLLLoss()


def computeTestSetAccuracy(model, loss_criterion):
    """
    Computes the accuracy and loss of the model on the test dataset.

    Parameters:
    model (torch.nn.Module): The trained model to evaluate.
    loss_criterion (torch.nn.Module): The loss function used for evaluation.
    """

    test_acc  = 0.0
    test_loss = 0.0

    with torch.no_grad():
        model.eval()

        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss    = loss_criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc += acc.item() * inputs.size(0)

            print(f"Test Batch number: {j:03d}, Test: Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}")

    avg_test_loss = test_loss / test_data_size
    avg_test_acc  = test_acc  / test_data_size

    print(f"\nAverage Test Loss: {avg_test_loss:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.4f}")


def predict(model, test_image_name):
    """
    Predicts the class of a given test image using a trained model.

    Parameters:
    model (torch.nn.Module): The trained model to use for prediction.
    test_image_name (str): The file path of the test image.
    """

    transform  = image_transforms['general']
    test_image = Image.open(test_image_name)

    plt.figure(figsize=(10, 7))
    plt.imshow(test_image)
    plt.axis('off')
    plt.show()

    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.cuda()

    with torch.no_grad():
        model.eval()
        out   = model(test_image_tensor)
        ps    = torch.exp(out)

        topk, topclass = ps.topk(3, dim=1)

        for i in range(3):
            print(f"Prediction {i+1}: {idx_to_class[topclass.cpu().numpy()[0][i]]}  "
                  f"Score: {topk.cpu().numpy()[0][i] * 100:.3f}%")


# ── Load model and run inference ──────────────────────────────────────────────
model = build_model(num_classes)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)

computeTestSetAccuracy(model, loss_func)

predict(model, '/home/kaushik/pytorch_bootcamp/assets/caltech256_subset/test/skunk/186_0062.jpg')