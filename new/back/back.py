from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
import uuid

app = FastAPI(title="YOLOv8 Object Detection API")

# Разрешаем доступ с любых источников
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель
model_path = "ml/best.pt"
model = YOLO(model_path)

# Папка для временных файлов
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"  # если нет расширения → ставим jpg
    temp_file = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"

    # читаем всё содержимое
    contents = await file.read()
    with open(temp_file, "wb") as f:
        f.write(contents)

    # делаем предсказание
    results = model(temp_file, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls),
                "confidence": float(box.conf),
                "box": box.xyxy[0].tolist()
            })

    return {"filename": file.filename, "detections": detections}
