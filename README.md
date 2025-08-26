# Détection des Équipements de Protection (Casques & Gilets) avec YOLOv8

## Description du projet
Ce projet a pour objectif de **détecter automatiquement les équipements de protection individuelle (EPI)** tels que :
- 🪖 **Casques de sécurité**
- 🦺 **Gilets de sécurité**

Il repose sur le framework **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**, un modèle d’intelligence artificielle spécialisé dans la détection d’objets en temps réel.

L’application permet :
1. **D’entraîner un modèle YOLOv8** personnalisé avec un dataset spécifique (images annotées de personnes avec/sans EPI).
2. **De tester la détection sur des images/vidéos** fournies par l’utilisateur.
3. **D’utiliser une webcam en direct** pour détecter les EPI en temps réel.

---


## Installation

### Cloner le projet

```bash
git clone https://github.com/Daouda-Ba/Detection-des-equipements-de-securite.git
cd Detection-des-equipements-de-securite
```

### Créer un environnement virtuel (optionnel mais recommandé)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

Le contenu de `requirements.txt` :

```txt
ultralytics
gradio
opencv-python
numpy
```

---

## Entraînement du modèle

Pour entraîner le modèle YOLOv8 avec vos propres données annotées :

```bash
yolo detect train data=data/data.yaml model=yolov8n.yaml epochs=50 imgsz=640
```

* `data.yaml` → contient la configuration des classes et chemins vers le dataset.
* `yolov8n.yaml` → architecture du modèle (nano YOLOv8, rapide et léger).
* `epochs=50` → nombre d’itérations.
* `imgsz=640` → taille des images d’entrée.

---

## Détection sur une image

Exécuter le script Python pour tester la détection :

```bash
python app.py
```

Extrait de `app.py` :

```python
from ultralytics import YOLO
import cv2

# Charger le modèle entraîné
model = YOLO("runs/detect/train/weights/best.pt")

# Charger une image
image = cv2.imread("test_image.jpg")

# Prédire
results = model.predict(image)

# Sauvegarder le résultat
results[0].save("output_image.jpg")
print("Résultat enregistré dans output_image.jpg")
```

---

## Détection en temps réel (Webcam)

```python
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Détection EPI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Résultats

* **Précision obtenue** : 81.5% (mAP\@50)

---

## Auteur

* **Daouda Ba** – Étudiant en Master Intelligence Artificielle
  Passionné par la vision par ordinateur et l’IA.

---
