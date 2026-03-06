import os
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from crnn import CRNN
from utils import decode

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DET_PATH = os.path.join(BASE_DIR, "weights", "best.pt")
CRNN_PATH = os.path.join(BASE_DIR, "weights", "ocr_crnn.pt")

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz-"
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(sorted(CHARS))}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models ONCE
det_model = YOLO(DET_PATH)

reg_model = CRNN(
    vocab_size=len(CHARS),
    hidden_size=256,
    n_layers=3,
)

reg_model.load_state_dict(torch.load(CRNN_PATH, map_location=device))
reg_model.to(device)
reg_model.eval()

transform = transforms.Compose([
    transforms.Resize((100, 420)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict(image_path):

    results = det_model(image_path)

    image = cv2.imread(image_path)
    output = []

    for r in results:

        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = map(int, box)

            # Crop cho CRNN
            cropped = image[y1:y2, x1:x2]
            pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

            img_tensor = transform(pil_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = reg_model(img_tensor).cpu()

            preds = logits.permute(1, 0, 2).argmax(2)
            text = decode(preds, IDX_TO_CHAR)[0]

            # 🔥 Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{text} {score:.2f}"
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            output.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "score": float(score)
            })

    # Convert BGR → RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, output