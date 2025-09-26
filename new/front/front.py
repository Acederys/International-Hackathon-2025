import streamlit as st
import requests
import base64
import io
import json

st.title("YOLOv8 + Anomaly Detection Demo")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg","jpeg","png","bmp","webp","tiff"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post("http://localhost:8000/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        st.json(result)

        if "annotated_image" in result:
            img_data = base64.b64decode(result["annotated_image"])
            st.image(io.BytesIO(img_data), caption="Annotated Image", use_container_width=True)

        st.write("Статус аномалии:", result["anomaly_text"])

        # Скачать JSON
        json_str = json.dumps(result, ensure_ascii=False, indent=4)
        st.download_button(
            label="Скачать JSON",
            data=json_str,
            file_name=f"{uploaded_file.name}_result.json",
            mime="application/json"
        )
    else:
        st.error(f"Ошибка API: {response.status_code}")
