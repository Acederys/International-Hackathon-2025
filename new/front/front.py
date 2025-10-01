import streamlit as st
import requests
import base64
import io
import json
import pandas as pd

st.title("YOLOv11 + OCR + Anomaly Demo")

# Чекбоксы
run_yolo = st.checkbox("YOLO (детекция инструментов)", value=True)
run_ocr = st.checkbox("EasyOCR (серийные номера)", value=False)
run_anomaly = st.checkbox("ResNet (аномалии)", value=False)

mode = st.radio("Режим загрузки", ["Одна картинка", "Несколько картинок"])

if mode == "Одна картинка":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg","jpeg","png","bmp","webp","tiff"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        params = {"run_yolo": run_yolo, "run_ocr": run_ocr, "run_anomaly": run_anomaly}

        response = requests.post("http://localhost:8000/predict", files=files, params=params)

        if response.status_code == 200:
            result = response.json()

            # === Таблица YOLO (комплектность инструментов) ===
            if run_yolo and result.get("completeness"):
                st.subheader("Комплектность инструментов")
                df = pd.DataFrame(list(result["completeness"].items()), columns=["Инструмент", "Статус"])

                def highlight(val):
                    if val == "ok":
                        return "background-color: lightgreen"
                    elif "missing" in val:
                        return "background-color: salmon"
                    elif "extra" in val:
                        return "background-color: khaki"
                    return ""

                st.dataframe(df.style.applymap(highlight, subset=["Статус"]))

            # === Аномалия ===
            if run_anomaly:
                st.subheader("Проверка на аномалии")
                if result.get("anomaly") == 1:
                    st.error("Аномалия обнаружена!")
                else:
                    st.success("Всё в порядке")

            # === Картинка ===
            if result.get("annotated_image"):
                img_data = base64.b64decode(result["annotated_image"])
                st.image(io.BytesIO(img_data), caption="Annotated Image", use_container_width=True)

            # === Скачать JSON ===
            json_str = json.dumps(result, ensure_ascii=False, indent=4)
            st.download_button(
                label="Скачать JSON",
                data=json_str,
                file_name=f"{uploaded_file.name}_result.json",
                mime="application/json"
            )
        else:
            st.error(f"Ошибка API: {response.status_code}")

else:  # Несколько картинок
    uploaded_files = st.file_uploader(
        "Загрузите изображения (несколько файлов)",
        type=["jpg","jpeg","png","bmp","webp","tiff"],
        accept_multiple_files=True
    )

    if uploaded_files:
        files = [("files", (f.name, f, f.type)) for f in uploaded_files]
        params = {"run_yolo": run_yolo, "run_ocr": run_ocr, "run_anomaly": run_anomaly}

        response = requests.post("http://localhost:8000/predict_batch", files=files, params=params)

        if response.status_code == 200:
            results = response.json()

            for result in results:
                st.markdown(f"### Результат: {result['filename']}")

                if run_yolo and result.get("completeness"):
                    df = pd.DataFrame(list(result["completeness"].items()), columns=["Инструмент", "Статус"])
                    st.dataframe(df)

                if run_anomaly:
                    if result.get("anomaly") == 1:
                        st.error("Аномалия обнаружена!")
                    else:
                        st.success("Всё в порядке")

                if result.get("annotated_image"):
                    img_data = base64.b64decode(result["annotated_image"])
                    st.image(io.BytesIO(img_data), caption="Annotated Image", use_container_width=True)

            # === Скачать общий JSON ===
            json_str = json.dumps(results, ensure_ascii=False, indent=4)
            st.download_button(
                label="Скачать все результаты JSON",
                data=json_str,
                file_name="batch_results.json",
                mime="application/json"
            )
        else:
            st.error(f"Ошибка API: {response.status_code}")
