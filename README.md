# YOLOv8 Classification Demo

Это демонстрационный сервис для классификации изображений с помощью **YOLOv8**. Сервис состоит из:

* **FastAPI** — API для предсказаний по загруженным изображениям.
* **Streamlit** — веб-интерфейс для загрузки изображений и просмотра результатов.
* **YOLOv8** — предобученная модель для классификации.

Модель уже включена в образ Docker, поэтому обучение на контейнере не требуется.

---

## Структура проекта

```
new/
├── back/
│   └── back.py             # FastAPI backend
├── front/
│   └── front.py            # Streamlit frontend
├── ml/
│   └── best.pt             # YOLOv8 модель
├── requirements.txt        # Python зависимости
├── Dockerfile
└── docker-compose.yml
```

---

## Требования

* Docker ≥ 24
* Docker Compose ≥ 2
* Windows / Linux / Mac (поддержка WSL на Windows)

---

## Сборка Docker-образа

Собираем образ с вашим сервисом:

```bash
docker-compose build detecot_tools
```

* Эта команда создаёт образ с FastAPI, Streamlit и вашей моделью.
* При необходимости зависимости кэшируются, чтобы ускорить сборку.

---

## Запуск сервиса

После сборки запускаем контейнер:

```bash
docker-compose up detecot_tools
```

* **FastAPI** будет доступен на: `http://localhost:8000`
* **Streamlit** будет доступен на: `http://localhost:8501`

Контейнер сразу использует модель `ml/best.pt` для предсказаний.

---

## Прекращение работы

Для остановки сервиса нажмите **Ctrl+C** или выполните:

```bash
docker-compose down
```

---

## Настройки

Если нужно изменить место хранения временных файлов Ultralytics, можно задать переменную окружения в `docker-compose.yml`:

```yaml
environment:
  - YOLO_CONFIG_DIR=/app/uploads
```

---

## Использование

1. Откройте веб-интерфейс Streamlit: `http://localhost:8501`
2. Загрузите изображение для классификации
3. Сервис отправит изображение на FastAPI, и вы увидите предсказанный класс и вероятности.
