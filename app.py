
import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

# Charger le modèle
model = YOLO("yolov8_trained.pt")

# Fonction pour traiter l'image
def detect_objects(image):
    image_np = np.array(image)
    results = model.predict(image_np)
    annotated_image = results[0].plot()
    return annotated_image

# Fonction pour traiter la vidéo
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (int(cap.get(3)), int(cap.get(4))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    return "output_video.mp4"

# Interface image
image_interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(label="Detected Safety Equipment"),
    title="Détection sur Image",
    description="Téléversez une image contenant des équipements de sécurité (casques, gilets, etc.)"
)

# Interface vidéo
video_interface = gr.Interface(
    fn=detect_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Video(label="Detected Video"),
    title="Détection sur Vidéo",
    description="Téléversez une vidéo pour détecter des équipements de sécurité"
)

# Interface avec onglets
iface = gr.TabbedInterface(
    [image_interface, video_interface],
    ["Image", "Vidéo"]
)

iface.launch()