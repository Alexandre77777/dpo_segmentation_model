import streamlit as st
import requests
from PIL import Image
import io
import base64
import numpy as np
import cv2

st.set_page_config(
    page_title="Сегментация спутниковых снимков",
    page_icon="🛰️",
    layout="wide"
)

st.title("Сегментация спутниковых снимков")

# URL бэкенда
BACKEND_URL = "http://localhost:8080"  # Замените на ваш URL после деплоя на render.com

def get_image_download_link(img, filename, text):
    """Создает ссылку для скачивания изображения"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

st.sidebar.header("Настройки")
patch_size = st.sidebar.slider("Размер патча", min_value=128, max_value=512, value=256, step=64)
subdivisions = st.sidebar.slider("Количество подразделений", min_value=2, max_value=4, value=2, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("""
## Классы
- 🟪 Здание (#3C1098)
- 🟣 Земля (#8429F6)
- 🔵 Дорога (#6EC1E4)
- 🟡 Растительность (#FEDD3A)
- 🟠 Вода (#E2A929)
- ⚪ Неразмеченный (#9B9B9B)
""")

uploaded_file = st.file_uploader("Выберите спутниковый снимок...", type=["jpg", "jpeg", "png"])

# Moved the button and processing logic above the columns
result_image = None
if uploaded_file is not None:
    # Moved button above columns
    process_button = st.button("Сегментировать изображение")
    
    if process_button:
        # Spinner now appears above columns
        with st.spinner("Обработка..."):
            # Подготавливаем файл для запроса
            files = {"file": ("image.jpg", uploaded_file.getvalue(), "image/jpeg")}
            params = {"patch_size": patch_size, "subdivisions": subdivisions}
            
            # Отправляем запрос на бэкенд
            try:
                response = requests.post(f"{BACKEND_URL}/predict/", files=files, params=params)
                response.raise_for_status()
                
                # Store result image for display in the column below
                result_image = Image.open(io.BytesIO(response.content))
                
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка соединения с бэкендом: {str(e)}")
            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Отображаем загруженное изображение
    image = Image.open(uploaded_file)
    col1.header("Исходное изображение")
    col1.image(image, caption="Загруженное изображение", use_column_width=True)
    
    # Display result if we have one
    if result_image is not None:
        # Создаем две колонки для заголовка и кнопки скачивания
        header_col, button_col = col2.columns([0.7, 0.3])
        header_col.header("Результат сегментации")
        button_col.markdown(get_image_download_link(result_image, "segmentation_result.png", "⬇️ Скачать"), unsafe_allow_html=True)
        
        col2.image(result_image, caption="Маска сегментации", use_column_width=True)
else:
    st.info("Загрузите изображение, чтобы начать")

# Добавляем информацию о модели
st.markdown("---")
st.markdown("""
## О модели
Это приложение использует глубокую нейронную сеть для семантической сегментации спутниковых снимков.
Модель может идентифицировать следующие объекты: здания, земля, дороги, растительность, вода и неразмеченные области.

### Как это работает
1. Загрузите спутниковый снимок
2. При необходимости настройте параметры
3. Нажмите "Сегментировать изображение"
4. Просмотрите и скачайте результат сегментации
""")
