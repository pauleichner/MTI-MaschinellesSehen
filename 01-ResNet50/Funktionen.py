import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


CATEGORIES = {
    0: 'BlackCrow',
    1: 'BEngulfing',
    2: 'Hammer',
    3: 'BHarami',
    4: 'MorningStar',
    5: 'ShootingStar',
    6: 'WhiteSoldier',
}

COLOR_MAP = {
    0: 'blue',    # BlackCrow
    1: 'green',   # BEngulfing
    2: 'red',     # Hammer
    3: 'cyan',    # BHarami
    4: 'magenta', # MorningStar
    5: 'yellow',  # ShootingStar
    6: 'orange',  # WhiteSoldier
}


# Trainingsfunktion
def train(epoch, model, train_loader, optimizer, criterion_class, device):
    model.train()
    for i, data in enumerate(train_loader, 0): # Iteration durch den DataLoader bekommt Bild und Label aus ResNet50Dataclass zurück
        # Bild und Labels auf GPU verschieben
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad() # Dmait sich die Gradienten nicht addieren
        # Vorhergesagte Klasse und Label | Form: ResNet50forChartPattern Klasse
        pred_label, pred_bbox = model(inputs) 
        # Trennung der Labels für Klassifizierung und Bounding Box
        true_label, true_bbox = labels[:, 0].long(), labels[:, 1:]
        # Berechnung der Verluste für beide Aufgaben
        loss_class = criterion_class(pred_label, true_label)
        loss_bbox = F.l1_loss(pred_bbox, true_bbox, reduction='mean') # Gäniger Verlust für BBoxen
        # Verluste addieren 
        loss = loss_class + loss_bbox
        loss.backward()
        optimizer.step()

        #print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {loss.item():.3f}')
        #print(f'Echte BBox: {bbox_labels[0].cpu().numpy()}')
        #print(f'Vorhergesagte BBox: {bbox_outputs[0].cpu().detach().numpy()}')


def test(model, test_loader, criterion_class, device, epoch):
    model.eval()
    total_loss = 0.0
    total_class_loss = 0.0
    total_bbox_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_iou = 0.0
    num_iou_calculations = 0

    # https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/
    with torch.no_grad(): # deaktiviiert Berechnung des Gradienten
        # Gleiches Vorgehen wie bei train()
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            class_outputs, bbox_outputs = model(images)
            class_labels, bbox_labels = labels[:, 0].long(), labels[:, 1:]
            loss_class = criterion_class(class_outputs, class_labels)
            loss_bbox = F.l1_loss(bbox_outputs, bbox_labels, reduction='mean')

            loss = loss_class + loss_bbox
            total_loss += loss.item()
            total_class_loss += loss_class.item()
            total_bbox_loss += loss_bbox.item()

            # Berechnen der Genauigkeit
            _, predicted = torch.max(class_outputs, 1)
            correct_predictions += (predicted == class_labels).sum().item()
            total_predictions += labels.size(0)

            # Berechnen der durchschnittlichen IoU für jede Vorhersage
            for j in range(labels.size(0)):
                # https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
                # Daten wieder auf die CPU verschieben da sonst die Berechnung zu einem np-Array nicht geht
                pred_bbox = bbox_outputs[j].cpu().numpy()
                true_bbox = bbox_labels[j].cpu().numpy()
                pred_bbox = convert_yolo_to_bbox(pred_bbox, [224, 224])
                true_bbox = convert_yolo_to_bbox(true_bbox, [224, 224])
                iou = calculate_iou(true_bbox, pred_bbox)
                total_iou += iou
                num_iou_calculations += 1

    avg_loss = total_loss / len(test_loader)
    avg_class_loss = total_class_loss / len(test_loader)
    avg_bbox_loss = total_bbox_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    avg_iou = total_iou / num_iou_calculations if num_iou_calculations > 0 else 0 # Da sonst Divison durch Null
    print(f'[INFO] EPOCH:{epoch + 1} -> Test Loss: {avg_loss:.3f}, Class Loss: {avg_class_loss:.3f}, BBox Loss: {avg_bbox_loss:.3f}, Accuracy: {accuracy:.3f}, Avg IoU: {avg_iou:.3f}')

    return avg_loss, avg_class_loss, accuracy, avg_iou


def convert_yolo_to_bbox(yolo_coords, img_dims):
    # https://stackoverflow.com/questions/56115874/how-to-convert-bounding-box-x1-y1-x2-y2-to-yolo-style-x-y-w-h
    # Nur Koordinaten der Bounding Box nehmen
    if len(yolo_coords) == 5:
        _, x_center, y_center, width, height = yolo_coords
    else:
        x_center, y_center, width, height = yolo_coords

    # Anpassung der Koordinaten von relativer Angabe auf absolute Pixelwerte
    x_min = (x_center - width / 2) * img_dims[1]  # Skalierung auf Bildbreite
    x_max = (x_center + width / 2) * img_dims[1]
    y_min = (y_center - height / 2) * img_dims[0]  # Skalierung auf Bildhöhe
    y_max = (y_center + height / 2) * img_dims[0]
    return [x_min, y_min, x_max, y_max]


def calculate_iou(box1, box2):
    # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # Berechnet die (x, y)-Koordinaten der Schnittflächen der Rechtecke
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Berechnung der Fläche von beiden Boxen
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Berechnet die IoU
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def display_images_with_boxes(test_loader, model, device, num_images=6):
    model.eval()
    # Wählt 6 random Bilder aus 
    images_to_display = random.sample(range(len(test_loader.dataset)), num_images)
    # 2x3 Subplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel() # Vereinfacht den Zugriff auf die Subplots

    for idx, img_idx in enumerate(images_to_display):
        # Iteration der Subplot Bilder
        image, label = test_loader.dataset[img_idx] # Läd das Bild, Klasse und BBox Koordinaten
        image_np = image.numpy().transpose(1, 2, 0)  # Konvertieren von PyTorch- zu NumPy-Format

        image = image.to(device)
        class_output, bbox_output = model(image.unsqueeze(0)) # Erhällt die Vorhersage des Bildes
        predicted_class = class_output.max(1)[1].item() # Maximum da das Label mit der höchsten Übereinstimmung
        predicted_bbox = bbox_output[0].cpu().detach().numpy()
        # In Pixelwerte umrechnen damit IoU berechnet werden kann
        true_bbox = convert_yolo_to_bbox(label.numpy()[1:], image_np.shape)
        pred_bbox = convert_yolo_to_bbox(predicted_bbox, image_np.shape)
        iou = calculate_iou(true_bbox, pred_bbox) # Berechnung der IoU
        true_label_name = CATEGORIES[int(label.numpy()[0])] # Namen des Labels bekommen
        predicted_label_name = CATEGORIES[predicted_class]  # Namen des Labels bekommen
        axs[idx].imshow(image_np)
        axs[idx].set_title(f'True: {true_label_name}, Pred: {predicted_label_name}, IoU: {iou:.2f}') # Titel anpassen

        # Echte Bounding Box in Farbe darstellen
        true_color = COLOR_MAP[int(label.numpy()[0])]
        rect_true = patches.Rectangle((true_bbox[0], true_bbox[1]), true_bbox[2] - true_bbox[0], true_bbox[3] - true_bbox[1], linewidth=2, edgecolor=true_color, facecolor='none')
        axs[idx].add_patch(rect_true)
        # Vorhergesagte Bounding Box in Farbe darstellen
        pred_color = COLOR_MAP[predicted_class]
        rect_pred = patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2] - pred_bbox[0], pred_bbox[3] - pred_bbox[1], linewidth=2, edgecolor=pred_color, facecolor='none', linestyle='--')
        axs[idx].add_patch(rect_pred)

    plt.tight_layout()
    plt.savefig('IoU_with_labels_and_colors.png')
    plt.show()


def plot_metrics(epochs, test_loss, class_loss, val_acc, val_iou):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, class_loss, label='Test Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Class Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_iou, label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Modell_Metriken.png')
    plt.show()

# https://stackoverflow.com/questions/61526287/how-to-add-correct-labels-for-seaborn-confusion-matrix
def plot_confusion_matrix(test_loader, model, device, num_classes):
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0], 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels[:, 0].cpu().numpy())

    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=CATEGORIES.values(), yticklabels=CATEGORIES.values())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix.png')
    plt.show()
