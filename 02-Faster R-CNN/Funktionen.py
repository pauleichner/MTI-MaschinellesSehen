import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import random
import numpy as np
from torchvision.ops import nms


COLOR_MAP = {
    1: 'red',
    2: 'green',
    3: 'blue',
    4: 'cyan',
    5: 'magenta',
    6: 'yellow',
    7: 'orange',
    0: 'black',
}

CATEGORIES = {
    1: 'BlackCrow', 
    2: 'BEngulfing',
    3: 'Hammer',
    4: 'BHarami',
    5: 'MorningStar',
    6: 'ShootingStar',
    7: 'WhiteSoldier',  
}



def train(epoch, model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0

    #n = 0
    for images, targets in train_loader:  # iamges: Batch von Tensoren | targets sind die Dicts mit den Angaben der BBoxen und Labels
        # print(f"Arbeite im Batch : {n} Bildgröße: {images[0].shape}")
        # n = n+1
        # Vorbereitung der Bilder, Anzahl je nach batch_size
        images_list = []
        for image in images:
            images_list.append(image.to(device))
            #print(images_list)
        # Für den jeweiligen Batch die Dicts laden
        targets_list = []
        for t in targets:
            target_dict = {}
            for k, v in t.items():
                    #print(f"Schlüssel: {k}, Wert: {v}") # Daten werden richtig geladen und ebenfalls mehrere Label angezeigt
                    target_dict[k] = v.to(device)
            targets_list.append(target_dict)
        
        # Berechne den Verlust, gibt das FRCNN direkt aus: https://datascience.stackexchange.com/questions/92309/how-to-interpret-fast-rcnn-metrics
        loss_dict = model(images_list, targets_list)
        # Berechnung der Gesamtverluste
        losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] + loss_dict.get('loss_objectness', 0) + loss_dict.get('loss_rpn_box_reg', 0)
        #print(losses)
        optimizer.zero_grad()
        losses.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # muss verwendet werden da sonst Gradienten zu stark ansteigen
        optimizer.step()
        running_loss += losses.item()

    average_loss = running_loss / len(train_loader)
    print(f"[INFO] EPOCH:{epoch} -> AVG Loss im Training : {average_loss}")
    return average_loss


def test(model, test_loader, device, iou_thresh=0.3):
    model.eval()
    metrics_summary = []
    all_pred_boxes = []
    all_true_boxes = []
    all_pred_labels = []
    all_true_labels = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images_list = []
            for image in images:
                image_on_device = image.to(device)
                images_list.append(image_on_device)
            outputs = model(images_list)

            for output, target in zip(outputs, targets): # Iteriert durch die Dicts
                # Konvertiere die Modellausgabe in CPU Tensoren für NMS
                cpu_output = {}
                for k, v in output.items():
                    cpu_output[k] = v.cpu() 
                    #print(cpu_output)
                # Wende NMS auf die Vorhersagen des aktuellen Bildes an, damit nicht so viele BBoxen geplottet werden
                nms_output = apply_nms(cpu_output, iou_thresh)
                # Vorhersagen und tatsächliche Labels und BBoxen extrahiert | die tatsächlichen Daten müssen zu erst wieder auf die CPU
                pred_boxes = nms_output['boxes'].numpy()
                pred_labels = nms_output['labels'].numpy()
                true_boxes = target['boxes'].cpu().numpy()
                true_labels = target['labels'].cpu().numpy()

                all_pred_boxes.append(pred_boxes)
                all_true_boxes.append(true_boxes)
                all_pred_labels.append(pred_labels)
                all_true_labels.append(true_labels)
                
                precision, recall, f1_score = calculate_precision_recall_f1(pred_boxes, true_boxes)
                metrics_summary.append((precision, recall, f1_score))
            # Zur Kontrolle während des Testens
            if i % 20 == 0:
                print(f"Batch {i} processed || pred label: {pred_labels} true label: {true_labels}")

    sum_precision = 0
    sum_recall = 0
    sum_f1_score = 0

    # Durch die Daten iterieren um die Summe bilden zu können
    for metrics in metrics_summary:
        sum_precision = sum_precision + metrics[0]  
        sum_recall = sum_recall + metrics[1]     
        sum_f1_score = sum_f1_score +  metrics[2]  

    # Berechne Durchschnittswerte
    avg_precision = sum_precision / len(metrics_summary)
    avg_recall = sum_recall / len(metrics_summary)
    avg_f1_score = sum_f1_score / len(metrics_summary)
    print(f"Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1-Score: {avg_f1_score}")

    return all_pred_boxes, all_true_boxes, all_pred_labels, all_true_labels, avg_precision, avg_recall, avg_f1_score


# https://pytorch.org/vision/main/generated/torchvision.ops.nms.html
def apply_nms(predictions, iou_threshold=0.3):
    keep = nms(predictions['boxes'], predictions['scores'], iou_threshold)
    return {
        'boxes': predictions['boxes'][keep],
        'labels': predictions['labels'][keep],
        'scores': predictions['scores'][keep],
    }


# Angelehnt an die Funktion welche Beim ResNet50 verwendet wird
def display_images_with_boxes(test_loader, model, device, num_images=6):
    model.eval()
    # Wählt 6 random Bilder aus
    images_to_display = random.sample(range(len(test_loader.dataset)), num_images)
    # 2x3 Subplot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.ravel()  # Vereinfacht den Zugriff auf die Subplots

    for idx, img_idx in enumerate(images_to_display):
        image, targets = test_loader.dataset[img_idx]
        image_np = image.numpy().transpose(1, 2, 0)  # Konvertierung für die Visualisierung

        image = image.to(device)
        with torch.no_grad():  # Keine Berechnung des Gradienten
            output = model([image])
        # Echte und Vorhergesagte Labels und BBox
        predicted_labels = output[0]['labels'].cpu().numpy()
        predicted_boxes = output[0]['boxes'].cpu().numpy()
        true_labels = targets['labels'].cpu().numpy()
        true_boxes = targets['boxes'].cpu().numpy()

        axs[idx].imshow(image_np)
        title = f'Image {img_idx}'

        # Echte Bounding Box in Farbe darstellen
        for box, label in zip(true_boxes, true_labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=COLOR_MAP[label], facecolor='none')
            axs[idx].add_patch(rect)
            title += f'Real: {CATEGORIES[label]}'
        # Vorhergesagte Bounding Box in Farbe darstellen
        for box, label in zip(predicted_boxes, predicted_labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor=COLOR_MAP[label], facecolor='none', linestyle='--')
            axs[idx].add_patch(rect)
        axs[idx].set_title(title)

    plt.tight_layout()
    plt.savefig('Predicted_BBOX.png')
    plt.show()


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


def calculate_precision_recall_f1(pred_boxes, true_boxes, iou_thresh=0.5):
    # https://www.linkedin.com/pulse/precision-recall-f1-score-object-detection-back-ml-basics-felix
    TP = 0 # true positive
    FP = 0 # false positive 
    FN = len(true_boxes)
    for pred_box in pred_boxes:
        for true_box in true_boxes:
            iou = calculate_iou(pred_box, true_box)
            if iou >= iou_thresh:
                TP = TP + 1
                FN = FN + 1
                break
        else:
            FP = FP + 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0 # Keine Division durch Null
    recall = TP / (TP + FN) if TP + FN > 0 else 0    # Keine Division durch Null
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1_score


def plot_results(epochs, avg_precisions, avg_recalls, avg_f1_scores):
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, avg_precisions, label='Average Precision')
    plt.title('Average Precision over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, avg_recalls, label='Average Recall')
    plt.title('Average Recall over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Recall')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, avg_f1_scores, label='Average F1-Score')
    plt.title('Average F1-Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average F1-Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Plots.png')
    plt.show()
