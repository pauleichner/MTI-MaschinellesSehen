import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ResNet50forChartPattern(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50forChartPattern, self).__init__()
        # Laden des ResNet50-Modells mit dem aktuellen 'Weights'-Parameter
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) # nicht mehr pretrained=True
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Identity
        self.resnet.fc = nn.Identity() # ersetzt die letzte Schicht, da sonst 1000 Klassen
        # Fully Connected Layer für die Klassen
        self.fc_class = nn.Linear(2048, num_classes)  # Anpassung für Klassifizierung
        # Fully Connected Layer für die Bounding Boxen 
        self.fc_bbox = nn.Linear(2048, 4)  # 4 Koordinaten für die Bounding Box

    def forward(self, x):
        x = self.resnet(x)
        # Ausgabe der Klasse
        class_output = self.fc_class(x)
        # Ausgabe der Bounding Box
        bbox_output = self.fc_bbox(x)
        return class_output, bbox_output