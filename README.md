# Projekt: Chartmustererkennung mit einem Convolutional Neural Network 
**Autor : Paul Leon Eichner**
## Einführung
### Einführung

In dem Folgenden Projekt wurde die Erkennung von Chartmustern mittels verschiedener Convolutional Neural Networks implementiert. 
Sogenannte Chartmuster können Aufschluss darüber geben wie sich der Kurs eines Vermögenswertes, beispielsweise einer Aktie, verhällt. Zusammen mit weiteren technischen und fundamentalen Indikatoren, können Chartmustern als zusätzlicher Indiz einer Preisbewegung gesehen werden.

Die übliche Erkennung von Chartmustern sieht keine Benutzung von Convolutional Neural Networks vor. Statdessen werden die Chartmuster durch Programmierung verschiedener mathemtischer Bedingungen erkannt und ausgewertet. Diese Erkennung ist in den meisten Fällen durchaus zutreffend und kann deswegen als Stand der Technik angesehen werden.
Obwohl die mathematische Analyse zur Erkennung von Chartmustern effektiv ist, kann dennoch die Erkennung druch CNN lohnenswert sein. Durch die Auswahl des geeigneten Datensatzes und die Erzielung hoher Genauigkeit kann man auf diese Weise das menschliche Verhalten bei der Identifizierung von Chartmuster nahezu wiederpsiegeln.
Dieser Ansatz hat das Potential komplexere Muster zu erkennen die möglicherweise von traditionellen Methoden übersehen werden.

### Related Work

Tatsächlich wurden bereits einige Projekte in diesem Umfeld implementiert. Ein Beispiel hierfür ist das
[YOLO Object Recognition Algorithm and ‘‘Buy-Sell Decision’’ Model Over 2D Candlestick Charts] von SERDAR BIROGUL, GÜNAY TEMÜR, AND UTKU KOSE welches am 23. April 2020 erschienen ist. In diesem Research Paper geht es um die Erkennung von Chartmustern mit Hilfe des YOLO-Netzwerks. Dabei wird das Netzwerk mit Hilfe historischer Daten auf bestimmte "Buy" oder "Sell" Chartmuster trainiert. Anschließend werden dann die gemachten Vorhersagen dafür benutzt Kauf- oder Verkaufsignale zu generieren. Aufgrund solider Ergebnisse dieses Research Papers wurde dieses Projekt im Rahmen des Mastermoduls "Maschinelles Sehen" an der Berliner Hochschule für Technik umgesetzt.


## Realisierung des Projektes
### Datensatzerstellung und Vorbereiten
Aufgrund der Schwirigkeit einen umfangreichen und genauen Datensatz für Chartmuster zu finden, wurde entschieden einen eigenen Datensatz zu erstellen. Dieser Schritt war notwendig da eine solider Datensatz eine wichtige Grundlage für ein erfolgreiches Training von CNN's ist. Zu diesem Zweck wurde ein Script geschrieben, welches auf die Erkennung und Speicherung von spezifischen Chartmustern in historischen Preisdaten ausgerichtet war. Die dabei verwendeten Daten stammen von verschiedenen Aktien aus dem "Standard & Poor's 500" Aktienindex und wurden  von der Yahoo Finance API zur Verfügung gestellt. Die dabei verwendete Zeiteinheit der Kerzen betrug eine Kerze pro Tag. 
Anschließend wurden die Bilder manuell gesichtet und die aussagekräftigesten und deutlichsten Muster für das Training gespeichert. 
Um die Chartmuster präzise zu Erkennung, wurde die Webanwendung CVAT(Computer Vision Annotation Tool) eingesetzt. Mit CVAT wurden die Bounding Boxen manuell um die identifizierten Chartmuster gezeichnet.
Folgenden Chartmuster wurden dabei berücksichtigt:
- Black Crow
- Bullish Engulfing
- Hammer
- Bullish Harami
- Morning Star
- Shooting Star
- White Soldier
Jedes der Chartmuster hat bestimmte Eigenschaften die auf potentielle Preisbewegungen hinweisen können.

Einige Beispiele der gezeichneten Bounding Boxen werden hier dargestellt:

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/e6a92b40-2f04-464e-b1ad-6656bde8c9d4)
![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/48585ffc-79d1-489d-81a2-79e120c646b8)
![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/8c044d57-2e5d-4e10-8746-acc0d96dc607)


Die zugehörigen Klassenlabel und Bounding Box Koordinaten wurden im YOLO-Format gespeichert. Dieses Format wurde gewählt, da es nicht auf absoluten Pixelwerten basiert sondern auf relativen Größen des Bildes. Dadurch wird die Skalierung der Bilder und die dadurch notwendige Umrechnung der Bounding Boxen stark vereinfacht.

```python
<object-class> <x_center> <y_center> <width> <height>
```

### ResNet50
Dieses Projekt zeigt die Implementierung eines Single Label Object Detection Modells unter Verwendung der vortrainierten ResNet50 Architektur mit PyTorch.
Die Grundlegende Projektstruktur sieht folgendermaßen aus:
```scss
main.py
ResNet50.py (Modell und Architektur)
DataSet.py (Datenvorbereitung und -verarbeitung)
Funktionen.py (Training, Test, Evaluierung, Visualisierung)
```
Bei der Implementierung des Netzwerkes waren die folgenden drei Schritte wichtig: 

1.) Erstellung der Datenklasse 

2.) Erstellung des Modellstruktur 

3.) Erstellung der Trainings- und Evaluierungsfunktion

#### Datenklasse
Bei der Erstellung der Datenklasse ist es "best practice" eine Klasse zu erstellen die von der PyTorch Klasse Dataset erbt. 
Die genaue Aufgabe der Klasse sind es für die von dem Modell geforderten Eingabewerte im richitgen Format bereitzustellen.
Bei der "ResNet50Dataclass" wird das Bild auf 224x224 Pixel transformiert und in die PyTorch eigene Datenart tensor umgewandelt. Die Entscheidung die Bilder auf 224x224 Pixel zu transformieren wurde bewusst getroffen, da das ResNet50 ursprünglich mit dem ImageNet Datensatz trainiert wurde, welcher ebenfalls Bilder dieser Größe verwendet hat. Diese Anpassung kann zu einer Performance-Steigerung führen, da zum einen die Verarbeitung von quadratischen Bilder für Netzwerke besser und zum anderen da natürlich eine kleinere Bildgröße weniger Speicherplatz auf der GPU in Anspruch nimmt. Zum Schluss wird das Bild und das dazugehörige Label als Tensor zurückgegeben.

#### Modellstruktur 
Die Klasse ResNet50forChartPattern definiert ein PyTorch-Modell das, wie oben beschrieben, auf der ResNet50 Modellarchitektur beruht und für das Training und die Evaluation auf einen eigenen Datensatz angepasst ist. Dazu wird die letzte Fully Connected Schicht des Netzwerks, mit Hilfe der nn.Identity()-Funktion, durch zwei eigene Layer ersetzt. Eine welche die Anzahl der vorhersagbaren Klassen: nn.Linear(2048, num_classes) und eine welche für die Ausgabe der Boudning Box zustädnig ist: nn.Linear(2048, 4). Im Forward-Pass wird nun der Eingabetensor "x" durch das Modell geleitet um Merkmale zu extrahieren und um anschließend durch die Fully Connected Layer Aussagen über die Klasse und die Bounding Box Koordinaten zu treffen.

```python
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
```

#### Trainings- und Evaluierungsfunktion
##### Trainingsfunktion
Die Trainingsfunktion trainiert das Modell für einen Epoch. Die Hauptaufgabe hierbei ist es aus den Eingabebilder sowohl die Klassen/Objekte als auch deren Position zu bestimmen. Diese Berechnung geschieht aufgrund der Möglichkeit von parallelisierten Arbeiten auf der GPU. Zunächst wird durch die Daten aus dem PyTorch DataLoader iteriert und je nach Batchgröße die Bilder und zugehörigen Labels bereitgestellt. Im Vorwärtsdurchlauf gibt das Modell Vorhersagen über Klasse und Position aus. Anschließend kommt es zur Verlustberechnung der beiden Vorhersagen. Dazu wird zum einen der Cross-Entropy-Loss verwendent, welcher typisch für Klassifierungsaufgaben ist. Zur Erfassung des Fehlers der vorhergesagten Bounding Boxen wird der L1-Verlust (Mean Absolute Error) verwendet. Dieser misst die durchschnittliche absolute Differenz zwischen den vorhergesagten und tatsächlichen Koordinaten.
Im Rückwärtsdurchlauf (Backpropagation) werden die Gradienten berechnet. Die Gradienten repräsentieren die partielle Ableitung der Verlustfunktion bezüglich der Gewichte und Biases des Netzwerks. Anschließend aktualisiert der Optimierer die Gewichte des Modells um den Fehler zu minimieren. In diesem Projekt wurde der SGD (Stochastic Gradient Descent) Optimierer mit L2-Reglularisierung verwendet. Die L2-Reglularisierung kann die Modellgenauigkeit erhöhen und soll gegen das Overfitting eines Modell wirken ,indem sie die Gewichte der Neuronen nicht zu groß werden lässt. Gerade bei kleineren Datensätzen kann das hilfreich sein.


##### Evaluierungsfunktion



### Faster RCNN



## Auswertung der Ergebnisse


## Zusammenfassung und Ausblick
