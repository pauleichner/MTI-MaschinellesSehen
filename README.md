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
Bei der "ResNet50Dataclass" wird das Bild auf 224x224 Pixel transformiert und in die PyTorch eigene Datenart tensor umgewandelt. Zum Schluss wird das Bild und das dazugehörige Label als Tensor zurckgegeben.




#### Modellstruktur 


#### Trainings- und Evaluierungsfunktion



### Faster RCNN



## Auswertung der Ergebnisse


## Zusammenfassung und Ausblick
