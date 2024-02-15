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

- Black Crow (Anzahl an Patterns: 132)
- Bullish Engulfing (Anzahl an Patterns: 185)
- Hammer (Anzahl an Patterns: 112)
- Bullish Harami (Anzahl an Patterns: 108)
- Morning Star (Anzahl an Patterns: 51)
- Shooting Star (Anzahl an Patterns: 92)
- White Soldier (Anzahl an Patterns: 100)

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
main.py (DataLoader, Ausführung)
ResNet50.py (Modellarchitektur)
DataSet.py (Datenvorbereitung und -verarbeitung)
Funktionen.py (Training, Test, Evaluierung, Visualisierung)
```
Bei der Implementierung des Netzwerkes waren die folgenden vier Schritte wichtig: 

1.) Erstellung der Datenklasse 

2.) Erstellung des Modellstruktur 

3.) Erstellung der Trainings- und Evaluierungsfunktion

4.) Erstellung weiterer Evaluierungsmetriken


#### Datenklasse
Bei der Erstellung der Datenklasse ist es "best practice" eine Klasse zu erstellen die von der PyTorch Klasse Dataset erbt. 
Die genaue Aufgabe der Klasse sind es für die von dem Modell geforderten Eingabewerte im richitgen Format bereitzustellen.
Bei der "ResNet50Dataclass" wird das Bild auf 224x224 Pixel transformiert und in die PyTorch eigene Datenart tensor umgewandelt. Die Entscheidung die Bilder auf 224x224 Pixel zu transformieren wurde bewusst getroffen, da das ResNet50 ursprünglich mit dem ImageNet Datensatz trainiert wurde, welcher ebenfalls Bilder dieser Größe verwendet hat. Diese Anpassung kann zu einer Performance-Steigerung führen, da zum einen die Verarbeitung von quadratischen Bilder für Netzwerke besser und zum anderen da natürlich eine kleinere Bildgröße weniger Speicherplatz auf der GPU in Anspruch nimmt. Jedoch darf die Bildgröße auch nicht zu klein werden, da sonst wichtige Features des Bildes verloren gehen. Zum Schluss wird das Bild und das dazugehörige Label als Tensor zurückgegeben.

#### Modellstruktur 
Die Klasse ResNet50forChartPattern definiert ein PyTorch-Modell das, wie oben beschrieben, auf der ResNet50 Modellarchitektur beruht und für das Training und die Evaluation auf einen eigenen Datensatz angepasst ist. Dazu wird die letzte Fully Connected Schicht des Netzwerks, mit Hilfe der nn.Identity()-Funktion, durch zwei eigene Layer ersetzt. Eine welche die Anzahl der vorhersagbaren Klassen: nn.Linear(2048, num_classes) und eine welche für die Ausgabe der Boudning Box zustädnig ist: nn.Linear(2048, 4). Im Forward-Pass wird nun der Eingabetensor "x" durch das Modell geleitet um Merkmale zu extrahieren und um anschließend durch die Fully Connected Layer Aussagen über die Klasse und die Bounding Box Koordinaten zu treffen.


#### Trainings- und Evaluierungsfunktion
##### Trainingsfunktion
Die Trainingsfunktion trainiert das Modell für einen Epoch. Die Hauptaufgabe hierbei ist es aus den Eingabebilder sowohl die Klassen/Objekte als auch deren Position zu bestimmen. Diese Berechnung geschieht aufgrund der Möglichkeit von parallelisierten Arbeiten auf der GPU. Zunächst wird durch die Daten aus dem PyTorch DataLoader iteriert und je nach Batchgröße die Bilder und zugehörigen Labels bereitgestellt. Im Vorwärtsdurchlauf gibt das Modell Vorhersagen über Klasse und Position aus. Anschließend kommt es zur Verlustberechnung der beiden Vorhersagen. Dazu wird zum einen der Cross-Entropy-Loss verwendent, welcher typisch für Klassifierungsaufgaben ist. Zur Erfassung des Fehlers der vorhergesagten Bounding Boxen wird der L1-Verlust (Mean Absolute Error) verwendet. Dieser misst die durchschnittliche absolute Differenz zwischen den vorhergesagten und tatsächlichen Koordinaten.
Im Rückwärtsdurchlauf (Backpropagation) werden die Gradienten berechnet. Die Gradienten repräsentieren die partielle Ableitung der Verlustfunktion bezüglich der Gewichte und Biases des Netzwerks. Anschließend aktualisiert der Optimierer die Gewichte des Modells um den Fehler zu minimieren. In diesem Projekt wurde der SGD (Stochastic Gradient Descent) Optimierer mit L2-Reglularisierung verwendet. Die L2-Reglularisierung kann die Modellgenauigkeit erhöhen und soll gegen das Overfitting eines Modell wirken ,indem sie die Gewichte der Neuronen nicht zu groß werden lässt. Gerade bei kleineren Datensätzen kann das hilfreich sein.

##### Evaluierungsfunktion
Ziel der Testfunktion ist es die Leistung des Modells zu bewerten. Dazu werden hier verschiedene Bewertungsmethoden eingesetzt. 

1.) Durchschnittlicher Gesamtverlust
Dabei handelt es sich um eine dimensionslose Größe die die wie folgt berechnet wird:

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/6dac7673-5972-4fec-b926-b111df791fac)

Dabei ist N die Länge des Test DataLoaders.


2.) Durchschnittlicher Klassifiezierungsverlust
Dabei handelt es sich um eine dimensionlose Größe die wie folgt berechnet wird:

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/34afeea2-704e-4a26-bc9f-afca116dd3a1)

3.) Genauigkeit
Die Genauigkeit ist eine prozentuale Größe die die richtigen Vorhersagen mit allen Vorhersagen setzt:

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/a7a0d7be-66f8-4a41-b650-fb931c6c72f3)

4.) Durchschnittlicher Intersection over Union
Die IoU ist eine dimensionsloses Verhältnis zwischen der "Ground Thruth" also der echten Bounding Box und der vorhergesagten Bounding Box mit möglichen Werten von 0 bis 1, wobei 0 -> keine Übereinstimmung bedeuten würde und 1 -> eine komplette Übereinstimmung bedeuten würde.

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/0db9e14c-2182-4278-85c6-b0587118212f)


Nachdem alle Funktionen implementiert wurden kann nun das Training über die main.py gestartet werden.
Der vollständige Code zu dieser Implementierung ist unter /01-ResNet50/ abgelegt. Der dazugehörige Datensatz liegt unter /03-Datensatz/

---

### Faster R-CNN
Nachdem die Implementierung eines Single Label Object Detectors besprochen und implementiert wurde, folgt nun die Implementierung eines Multi Label Object Detectors, also der Lokalisierung von mehr als einem Objekt innerhalb desselben Bildes. Dies ist von Vorteil da einige der Bilder des Datensatzes ebenfalls mehrere Chartmuster behinhalten (siehe Beispiel).

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/43c27e44-f444-4c36-8b81-dec53a0e02f7)

#### Faster R-CNN vs YOLO
Um eine Entscheidung zu treffen welches Modell besser geeignet ist, müssen beide Modelle auf ihre Stärken und Schwächen untersucht werden. 
Das Faster R-CNN verwendet ein Region Proposal Network um Objektkandidaten vorzuschlagen. Auf diesen Vorschlag werden dann Klassifizierer und Bounding Box Regressor angewendet. Dieser zweistufige Prozess führt in der Regel zu einer höheren Genauigkeit bedeutet aber auch, dass Training des Modells und Vorhersagen länger dauern.
Das YOLO-Netzwerk betrachtet auf der anderen Seite die Objekterkennung als ein einzelnen Regressionsporblem, dass von der Bildpixel direkt zu Bounding-Box Koordinaten und Klassenvorhersagen führt. Dieser Ansatz ist deutlich schneller als der des F-R-CNN kann aber in einigen Fällen ungenauer Ausfallen.
Da das Projekt im Bereich des Swing-Tradings angewendet werden soll, also in einem "langsameren" Handelsbereich, wo Positionen über mehrere Tage und Wochen gehalten werden können, fiel die Wahl au das Faster R-CNN. Hierbei werden Genauigkeit und Aussagekraft der Vorhersagen wichtiger bewertet als die Schnelligkeit.

#### Projektimplementierung
Da sich der grundlegende Aufbau des Projekts zu dem des Projekts für das ResNet50 nicht wesentlich unterscheided, wird hier nur auf die Unterschiede eingegangen.
Die grobe Projektstrukutur sieht folgendermaßen aus:
```scss
main.py (Modell, DataLoader, Ausführung)
DataSet.py (Datenvorbereitung und -verarbeitung)
Funktionen.py (Training, Test, Evaluierung, Visualisierung)
```


##### Datenklasse
Die Bereitstellung von Daten für das F-R-CNN Modell erforderte einige spezifische Änderungen im Vergleich zu der Datenklasse des ResNet50. Ein entscheidender Unterschied liegt in der Art und Weise wie das Ziel (oder target) definiert wird, das für jede Trainings oder Testinstanz ein Dictionary anstelle eines einfachen Labels ist. Diese Dictionary enthällt Informationen über die Bounding Boxes, Klassenbezeichnungen, Bild-ID's und Fläche der Bounding Boxen. Hierbei ist zu beachten, dass die Bounding Boxes als absolute Pixelwerte angegeben werden müssen. Um dies zu erreichen muss eine Umrechnung von YOLO-Koordinaten zu absoluten Pixelwerten stattfinden. Diese sind in der Regel so dargestellt:

```python
<x_min> <y_min> <x_max> <y_max>
```

Dabei geschieht die Umrechnung folgendermaßen:

![image](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/a7c9852b-27ee-44c1-a348-b881fdf21c92)

Ein weiterer Punkt der beachtet werde musste, ist das das F-R-CNN standardmäßig das Klassenlabel "0" als Label für den Hintergrund sieht. Daher mussten hier die Klassenindizies um eins nach vorne geshiftet werden.

```python
with open(label_path) as f:
    for line in f.readlines():
        class_id, x_center, y_center, width, height = [float(x) for x in line.strip().split()]
        labels.append(int(class_id) + 1)
```

#### Modellstruktur
Das Modell des Faster R-CNN kann wie folgt implementiert werden.
```python
def create_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    # Anzahl der in_features
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 für den Hintergrund
    return model
```
Auch hier wird erneut die letzte Schicht 
```python
model.roi_heads.box_predictor
```
auf die Klassen des eigenen Datensatzes angepasst:
```python
FastRCNNPredictor(in_features, num_classes + 1)
```
Auch hier muss der Hintergrund, als eigene Klasse, berücksichtigt werden.

#### Trainings- und Evaluierungsfunktion
##### Trainingsfunktion
Im Wesentlichen unterscheided sich die Trainingsfunktion nur kaum, in Hinblick auf den grundlegenden Aufbau, zu der des ResNet50. Lediglich die Verarbeitung der Bilder und der Daten des Train Loaders sind, aufgrund der Struktur der Eingabe die das F-R-CNN fordert, verschieden. Auch bei der Verlustberechnung ist durch die Vewendung des F-R-CNN Modells einiges anders. Hier werden, je nachdem in welchem Modus (.train() oder .eval()), sich das Modell befindet, die verschiedenen Verluste im Ouptut Dictionary gespeichert.
Eine kleine Übersicht liefert die Ausgabe des Outputs auf die Konsole welches auf der folgenden Seite gemacht wurde [].
Hier ist ein kleines Beispiel von der Internetseite zur Verdeutlichung:
``` python
Epoch: [6]  [ 50/119]  eta: 0:00:46  lr: 0.000050  loss: 0.3973 (0.4123)  loss_classifier: 0.1202 (0.1248)  loss_box_reg: 0.1947 (0.2039)  loss_objectness: 0.0315 (0.0366)  loss_rpn_box_reg: 0.0459 (0.0470)  time: 0.6730  data: 0.1297  max mem: 3105
```
Aufgrund der gelieferten Ausgaben des Systems kann der Gesamtverlust berechnet werden:
```python
loss_dict = model(images_list, targets_list)
losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] + loss_dict.get('loss_objectness', 0) + loss_dict.get('loss_rpn_box_reg', 0)
```
Ein weiterer Unterschied welcher aber erst während des Trainings implementiert wurde ist die Begrenzung des Gradienten durch:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Dies war notwending da während anfänglicher Trainingstyklien es zu einem zu hohen Gradienten kam und das Modell darauf die Anpassung der Gewichte nicht mehr ordnungsgemäß anpassen konnte. Grund dafür ist das gerade bei tiefen Netzen wie dem Faster R-CNN es durch wiederholtes Anwenden der Kettenregel während der Backpropagation zu hohen Gradienten kommen kann, da jede Schicht zur Steigerung des Gradienten beiträgt. Ist die Steigung Größe der Steigung größer als 1 kann es schnell zu einem exponentiellen Anstieg kommen durch den das Modell nicht mehr in der Lage ist die Gewichte ordnungsgemäß anzupassen.

Dieser Plot wurde während des Trainings des Modells aufgenommen:
![Plots](https://github.com/pauleichner/MTI-MaschinellesSehen/assets/77249319/bd862e33-7b81-479c-acf5-d2b5c320a0f6)



## Auswertung der Ergebnisse

#### Evaluierungsmetriken
Da es ohne 



## Zusammenfassung und Ausblick
