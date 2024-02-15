import torch
import torch.nn as nn
from torchvision.ops import nms
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import os
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

# Eigene Funktionen
from DataSet import FRCNNDataclass
from Funktionen import train, test, display_images_with_boxes, plot_results

########################## Eingaben ##########################
# Muss nach eins nach vorne geshiftet werden da FRCNN standardmäßg 0 als Hintergrund ansieht
category_colors = {
    1: 'BlackCrow', 
    2: 'BEngulfing',
    3: 'Hammer',
    4: 'BHarami',
    5: 'MorningStar',
    6: 'ShootingStar',
    7: 'WhiteSoldier',  
}

# Hyperparameter
Pixel = 224
learning_rate = 1e-3
epochs = 10
batch_size = 2

# Modell erstellen https://www.kaggle.com/code/maherdeebcv/pytorch-faster-r-cnn-with-resnet152-backbone <- Line:19 nur hier vortrainiertes Modell
def create_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    # Anzahl der in_features
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 für den Hintergrund
    return model

def get_transform():
    def transform(image, target):
        image = F.resize(image, [Pixel, Pixel]) # auf kleiner Pixelgröße skalieren und quadratisch machen
        image = F.to_tensor(image)
        return image, target
    return transform

# Datensätze und DataLoader erstellen
train_dataset = FRCNNDataclass('train/images', 'train/labels', transform=get_transform())
test_dataset = FRCNNDataclass('test/images', 'test/labels', transform=get_transform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))) # https://github.com/pytorch/pytorch/issues/42654
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Modell erstellen
model = create_model(len(category_colors))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO]: {device} wird verwendet")
model.to(device)
# Hier den Optimizer Adam benutzt, da davor mit SGD die Ergebnisse nicht gut waren...
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # L2-Regularisierung

########################## Training und Test ##########################
avg_precisions = []
avg_recalls = []
avg_f1_scores = []
for epoch in range(epochs):
    train_loss = train(epoch, model, train_loader, optimizer, device)
    pred_boxes, true_boxes, pred_labels, true_labels, avg_precision, avg_recall, avg_f1_score = test(model, test_loader, device)
    #print(f"Predicted Boxes: {pred_boxes[5]} |  True Boxes: {true_boxes[5]}")
    avg_precisions.append(avg_precision)
    avg_recalls.append(avg_recall)
    avg_f1_scores.append(avg_f1_score)
print('Finished Training')
########################## Speichern des Modells ##########################
model_path = 'resnet50_adapted_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Modell gespeichert unter {model_path}")

########################## Visualisierung ##########################
display_images_with_boxes(test_loader, model, device, num_images=6)
plot_results(epochs, avg_precisions, avg_recalls, avg_f1_scores)









