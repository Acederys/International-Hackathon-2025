# YOLOv8 Tools + Anomaly Detection Demo

Это демонстрационный сервис для анализа комплектности инструментов и выявления аномалий на изображениях с помощью YOLOv8 и ResNet18. Сервис состоит из:

* **FastAPI** — API для предсказаний по загруженным изображениям.
* **Streamlit** — веб-интерфейс для загрузки изображений и просмотра результатов.
* **YOLOv8** — модель для детекции инструментов.
* **ResNet18** — модель для выявления аномалий на изображении.
* **EasyOCR** — распознавание серийных номеров/меток на инструментах.

Модели уже включены в образ Docker, поэтому обучение на контейнере не требуется.

---

## Структура проекта

```
new/
├── back/
│   └── back.py             # FastAPI backend
├── front/
│   └── front.py            # Streamlit frontend
├── ml/
│   ├── best2.pt            # YOLOv8 модель инструментов
│   └── model_resnet18.pth  # Модель аномалий (ResNet18, 2 класса)
├── uploads/                # Временные файлы
├── requirements.txt        # Python зависимости
├── Dockerfile
└── docker-compose.yml
```

---

## Возможности

1. **Детекция инструментов** с YOLOv8:

   * Вывод классов и вероятностей.
   * Отмеченные bounding box на изображении.
   * OCR-считывание серийных номеров на инструментах.
   * Проверка комплектности набора инструментов.

2. **Аномалия на изображении** с ResNet18:

   * 0 = всё в порядке
   * 1 = аномалия обнаружена

3. **JSON вывод**:

   * `detections` — детекции инструментов с координатами, OCR, вероятностью.
   * `completeness` — статус комплектности.
   * `anomaly` — бинарный результат аномалии.
   * `anomaly_text` — человекочитаемый результат.
   * `annotated_image` — base64 изображения с bounding box.

---

## Требования

* Docker ≥ 24
* Docker Compose ≥ 2
* Windows / Linux / Mac (поддержка WSL на Windows)

---

## Сборка Docker-образа

```bash
docker-compose build detecot_tools
```

> Образ содержит FastAPI, Streamlit, YOLOv8 и ResNet18.

---

## Запуск сервиса

```bash
docker-compose up detecot_tools
```

* **FastAPI**: `http://localhost:8000`
* **Streamlit**: `http://localhost:8501`

Контейнер сразу использует модели `ml/best2.pt` и `ml/model_resnet18.pth` для предсказаний.

---

## Использование через веб-интерфейс

1. Откройте `http://localhost:8501`.
2. Загрузите изображение.
3. Streamlit покажет:

   * исходное изображение,
   * JSON с детекциями, комплектностью и результатом аномалии,
   * аннотированное изображение с bounding box.
4. JSON можно скачать кнопкой «Скачать JSON».

---

## Использование **только API** (без Streamlit)

Можно отправлять POST-запрос на FastAPI:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/путь/к/изображению.jpg"
```

Пример на Python:

```python
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8000/predict"
files = {"file": open("example.jpg", "rb")}
resp = requests.post(url, files=files)

data = resp.json()
print(data["anomaly_text"])
print(data["completeness"])
print(data["detections"])

# Для отображения аннотированного изображения
img_data = base64.b64decode(data["annotated_image"])
img = Image.open(BytesIO(img_data))
img.show()
```

---

## Настройки

Если нужно изменить место хранения временных файлов Ultralytics:

```yaml
environment:
  - YOLO_CONFIG_DIR=/app/uploads
```

---

## Остановка сервиса

```bash
docker-compose down
```
