import streamlit as st
import requests

st.title("YOLOv8 Classification Demo")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # показываем картинку
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # отправляем на FastAPI
    files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        st.write("**Предсказанный класс:**", result["predicted_class"])
        st.write("**Вероятности:**", result["probabilities"])
    else:
        st.error("Ошибка API")
