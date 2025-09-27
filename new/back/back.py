
import torch
import torchvision.transforms as T
from torchvision import models
from torch import nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
import uuid
from collections import Counter
import cv2
import numpy as np
import easyocr
import re
import base64
from fastapi.responses import JSONResponse

app = FastAPI(title="YOLOv8 Tools + Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === YOLO ===
model_yolo = YOLO(str(Path(__file__).parent.parent / "ml" / "best2.pt"))

# === OCR ===
reader = easyocr.Reader(['ru', 'en'])
pattern = re.compile(r'^[A-ZА-Я]{2}-\d{6}-(?:[1-9]|[1-4][0-9]|50)$')

# Эталонный комплект
REFERENCE_TOOLS = {
    "1": {"name": "Отвертка «-»", "expected_count": 1},
    "2": {"name": "Отвертка «+»", "expected_count": 1},
    "3": {"name": "Отвертка на смещенный крест", "expected_count": 1},
    "4": {"name": "Коловорот", "expected_count": 1},
    "5": {"name": "Пассатижи контровочные", "expected_count": 1},
    "6": {"name": "Пассатижи", "expected_count": 1},
    "7": {"name": "Шэрница", "expected_count": 1},
    "8": {"name": "Разводной ключ", "expected_count": 1},
    "9": {"name": "Открывашка для банок с маслом", "expected_count": 1},
    "10": {"name": "Ключ рожковый накидной ¾", "expected_count": 1},
    "11": {"name": "Бокорезы", "expected_count": 1}
}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# === Аномалия (ResNet18) ===
device = "cuda" if torch.cuda.is_available() else "cpu"
anomaly_model_path = Path(__file__).parent.parent / "ml" / "model_resnet18.pth"

anomaly_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_feats = anomaly_model.fc.in_features
anomaly_model.fc = nn.Linear(num_feats, 2)  # 0=normal, 1=anomaly
anomaly_model.load_state_dict(torch.load(anomaly_model_path, map_location=device))
anomaly_model = anomaly_model.to(device)
anomaly_model.eval()

transform = T.Compose([
    T.Resize((768, 768)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# === Вспомогательные функции ===
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    return cv2.warpAffine(img, M, (nW, nH))

def ocr_until_match(img):
    for angle in range(0, 360, 10):
        rotated = rotate_image(img, angle)
        results = reader.readtext(rotated)
        for (bbox, text, conf) in results:
            text_clean = text.strip().replace(" ", "").upper()
            if pattern.match(text_clean):
                return {"text": text_clean, "confidence": round(conf, 4), "angle": angle}
    return None

def predict_anomaly(img_cv2):
    pil_img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    x = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = anomaly_model(x)
        label = pred.argmax(1).item()
    return label  # 0=normal, 1=anomaly

# === API ===
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    run_yolo: bool = Query(True, description="Запустить YOLO детекцию"),
    run_ocr: bool = Query(False, description="Запустить OCR"),
    run_anomaly: bool = Query(False, description="Запустить аномалию")
):
    ext = Path(file.filename).suffix or ".jpg"
    temp_file = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

    contents = await file.read()
    with open(temp_file, "wb") as f:
        f.write(contents)

    img = cv2.imread(str(temp_file))

    detections = []
    completeness = {}
    anomaly_label, anomaly_text = None, None
    img_b64 = None

    # YOLO + OCR
    if run_yolo:
        results = model_yolo(temp_file, verbose=False)
        class_counts = Counter()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                cls_name = model_yolo.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                crop_img = img[y1:y2, x1:x2]
                match = None
                if run_ocr:
                    match = ocr_until_match(crop_img)

                class_counts[cls_name] += 1
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2],
                    "ocr_text": match["text"] if match else None,
                    "ocr_confidence": match["confidence"] if match else None,
                    "ocr_angle": match["angle"] if match else None
                })

        # Комплектность
        for tool_id, tool_info in REFERENCE_TOOLS.items():
            name = tool_info["name"]
            expected_count = tool_info["expected_count"]
            detected_count = class_counts.get(tool_id, 0)
            if detected_count == expected_count:
                completeness[name] = "ok"
            elif detected_count < expected_count:
                completeness[name] = f"missing {expected_count - detected_count}"
            else:
                completeness[name] = f"extra {detected_count - expected_count}"

        # Аннотированное изображение
        annotated_img = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Аномалия
    if run_anomaly:
        anomaly_label = predict_anomaly(img)
        anomaly_text = "Аномалия обнаружена!" if anomaly_label == 1 else "Все в порядке"

    return JSONResponse({
        "filename": file.filename,
        "detections": detections if run_yolo else None,
        "completeness": completeness if run_yolo else None,
        "anomaly": anomaly_label if run_anomaly else None,
        "anomaly_text": anomaly_text if run_anomaly else None,
        "annotated_image": img_b64 if run_yolo else None
    })
