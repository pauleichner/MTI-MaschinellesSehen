import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ResNet50Dataclass(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.images = []
        # Durch alle Bilder iterieren
        for img in os.listdir(images_path):
            if img.lower().endswith('.png'): # Wenn png Endung dnn Name des Bildes in die Liste packen
                self.images.append(img)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, self.images[idx])
        # Im Labels Ordner suchen nach den gleichen Namen suchen nur Endung jetzt .txt
        label_name = os.path.join(self.labels_path, os.path.splitext(self.images[idx])[0] + '.txt')
        # https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/ -> Konvertierung in RGB wird in jedem Projekt machen
        image = Image.open(img_name).convert('RGB')
        # Label laden
        try:
            with open(label_name, 'r') as f:
                label = f.readline().split()
                label = torch.tensor([float(i) for i in label])  # Einschließlich des Klassenindex
        except FileNotFoundError:
            print(f"Label-Datei nicht gefunden: {label_name}")
            return None
        
        if self.transform:
            image = self.transform(image)
        # Rückgabe: Tensor(transformiertes Bild) , erste Zeile der txt. Datei im Yolo Format -> 1 0.562972 0.357490 0.080479 0.257427
        return image, label