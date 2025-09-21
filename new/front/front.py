import streamlit as st
import requests

st.title("YOLOv8 Object Detection Demo")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"])

if uploaded_file is not None:
    # показываем загруженное изображение
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # отправляем файл на API
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post("http://localhost:8000/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        st.json(result)
    else:
        st.error(f"Ошибка API: {response.status_code}")
