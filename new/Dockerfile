FROM python:3.10-slim

WORKDIR /app

# Сначала копируем только requirements.txt для кэширования зависимостей
COPY requirements.txt .

# Устанавливаем системные библиотеки для OpenCV и Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Теперь копируем весь проект (включая ml/best.pt)
COPY . .

# Порты
EXPOSE 8000 8501

# Создаём папку для временных файлов
RUN mkdir -p /app/uploads

# Запускаем бэкенд и фронтенд одновременно
CMD ["sh", "-c", "uvicorn back.back:app --host 0.0.0.0 --port=8000 & streamlit run front/front.py --server.port=8501 --server.address=0.0.0.0"]
