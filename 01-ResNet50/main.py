import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# Eigene Funktionen: 
from DataSet import ResNet50Dataclass
from ResNet50 import ResNet50forChartPattern
from Funktionen import train, test, convert_yolo_to_bbox, calculate_iou, display_images_with_boxes, plot_metrics, plot_confusion_matrix
########################## Eingaben ##########################
CATEGORIES = {
    0: 'BlackCrow',
    1: 'BEngulfing',
    2: 'Hammer',
    3: 'BHarami',
    4: 'MorningStar',
    5: 'ShootingStar',
    6: 'WhiteSoldier',
}
category_colors = {
    0: 'blue',    # BlackCrow
    1: 'green',   # BEngulfing
    2: 'red',     # Hammer
    3: 'cyan',    # BHarami
    4: 'magenta', # MorningStar
    5: 'yellow',  # ShootingStar
    6: 'orange',  # WhiteSoldier
}
# Hyperparameter
num_classes = len(CATEGORIES)
Height = 482
Width = 794
learning_rate = 1e-3
num_epoch = 2000
batch_size = 32
momentum = 0.9


########################## Datenvorbereitung ##########################
# Transformationen
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)) # Da das ResNet50 auf Bildern der Größe trainiert wurde
])
# Datensätze und PyTorch DataLoader
train_dataset = ResNet50Dataclass('train/images', 'train/labels', transform)
test_dataset = ResNet50Dataclass('test/images', 'test/labels', transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model festlegen
model = ResNet50forChartPattern(num_classes=num_classes)
# Gerät festlegen
device = torch.device("cuda")
print(f"[INFO]: {device} wird verwendet")
# Modell auf CUDA verschieben
model.to(device)
# Verluste und Optimierer definieren
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)  #L2-Regularisierung


########################## Training und Test ##########################
test_losses = []
test_class_losses = []
test_accuracies = []
avg_ious = []

for epoch in range(num_epoch):
    train(epoch, model, train_loader, optimizer, criterion, device)
    test_loss, test_class_loss, test_accuracy, avg_iou = test(model, test_loader, criterion, device, epoch)
    # Werte in Listen speichern
    test_losses.append(test_loss)
    test_class_losses.append(test_class_loss)
    test_accuracies.append(test_accuracy)
    avg_ious.append(avg_iou)
print('Finished Training')


########################## Speichern des Modells ##########################
model_path = 'resnet50_adapted_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Modell gespeichert unter {model_path}")

########################## Visualisierung ##########################
display_images_with_boxes(test_loader, model, device)
epochs_array = list(range(1, num_epoch + 1))
plot_metrics(epochs_array, test_losses, test_class_losses, test_accuracies, avg_ious)
plot_confusion_matrix(test_loader, model, device, num_classes)


