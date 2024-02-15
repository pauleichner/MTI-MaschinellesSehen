import os
import torch
from PIL import Image
from torch.utils.data import Dataset

Pixel = 224 # Da das ResNet50 als Backbone für das FRCNN genutzt wird
class FRCNNDataclass(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images_path = images_path
        self.labels_path = labels_path
        self.transform = transform
        self.images = []
        for img in os.listdir(images_path):
            if img.lower().endswith('.png'):
                self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.images[idx])
        label_path = os.path.join(self.labels_path, self.images[idx].replace('.png', '.txt'))
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = [float(x) for x in line.strip().split()]
                labels.append(int(class_id) + 1)  # Shift um eins nach oben, da FRCNN die 0 als Hintergrund animmt
                # Umrechnen von YOLO zu absoluten Koordinaten [xmin, ymin, xmax, ymax]
                # Auch wenn die Transformation erst danach durchgeführt wird, kann die BBOX in absoluten Werten schon gemacht werden, da die Größe klar ist
                img_width, img_height = Pixel, Pixel
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64) # wird hier immer auf Null gesetzt
        # https://towardsdatascience.com/everything-about-fasterrcnn-6d758f5a6d79#:~:text=Fasterrcnn%20expects%20our%20data%20in,we%20have%20our%20images%20in.
        # https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html#torchvision.models.detection.fasterrcnn_resnet50_fpn
        # https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch
        target = {}  #FRCNN erwartet es als Dict mit den Folgenden Werten:
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            image, target = self.transform(image, target)

        #print(image.shape)
        return image, target