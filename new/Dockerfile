FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные библиотеки
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Сначала зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- EasyOCR ----
# Кладём подготовленную папку .EasyOCR с весами внутрь образа
RUN mkdir -p /root/.EasyOCR
COPY .EasyOCR /root/.EasyOCR

# ---- YOLO ----
RUN mkdir -p /root/.cache/ultralytics
# копируем свои веса
COPY ml/best2.pt /app/ml/best2.pt
# копируем дефолтные yolov8n.pt чтобы Ultralytics не качал
COPY ml/yolov8n.pt /root/.cache/ultralytics/yolov8n.pt

# Теперь копируем весь проект
COPY . .

# Папка для аплоадов
RUN mkdir -p /app/uploads

# Открываем порты
EXPOSE 8000 8501

# Запускаем backend и frontend
CMD ["sh", "-c", "uvicorn back.back:app --host 0.0.0.0 --port=8000 & streamlit run front/front.py --server.port=8501 --server.address=0.0.0.0"]
