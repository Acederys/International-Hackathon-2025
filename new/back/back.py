from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pathlib import Path
import shutil
import uuid

app = FastAPI(title="YOLOv8 Classification API")

# разрешаем доступ с любых источников (для Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# загружаем модель
model_path = r"ml/best.pt"
model = YOLO(model_path)

# временная папка для загруженных изображений
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # сохраняем файл
    ext = Path(file.filename).suffix
    temp_file = UPLOAD_DIR / f"{uuid.uuid4()}{ext}"
    with temp_file.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # делаем предсказание
    result = model(temp_file, verbose=False)[0]  # predictions for single image
    probs = result.probs.data.cpu().numpy()
    pred_class = int(probs.argmax())

    return {
        "filename": file.filename,
        "predicted_class": pred_class,
        "probabilities": probs.tolist()
    }
