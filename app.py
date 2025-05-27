import streamlit as st
import requests
from PIL import Image
import io
import base64

# 1. Конфигурация страницы
st.set_page_config(
    page_title="Сегментация спутниковых снимков",
    page_icon="🛰️",
    layout="wide"
)

# 2. URL вашего бэкенда
BACKEND_URL = "https://dpo-segmentation-model.onrender.com"

# 3. Вспомогательные функции
def get_image_download_link(img: Image.Image, filename: str, text: str) -> str:
    """Генерирует HTML-ссылку для скачивания PIL-изображения."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'

def call_segmentation_api(file_bytes: bytes, patch_size: int, subdivisions: int) -> Image.Image:
    """Отправляет запрос на бэкенд и возвращает результат как PIL.Image."""
    files = {"file": ("image.png", file_bytes, "image/png")}
    params = {"patch_size": patch_size, "subdivisions": subdivisions}
    resp = requests.post(f"{BACKEND_URL}/predict/", files=files, params=params)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

# 4. Сайдбар: настройки и легенда
with st.sidebar:
    st.header("⚙️ Параметры сегментации")
    with st.form("settings"):
        patch_size   = st.slider("Размер патча", 128, 512, 256, step=64)
        subdivisions = st.slider("Подразделений", 1, 4, 2)
        submit = st.form_submit_button("Применить")
    st.markdown("---")
    with st.expander("📋 Легенда классов", expanded=True):
        classes = {
            "Здание":        "#3C1098",
            "Земля":         "#8429F6",
            "Дорога":        "#6EC1E4",
            "Растительность":"#FEDD3A",
            "Вода":          "#E2A929",
            "Неразмеченное": "#9B9B9B",
        }
        for name, color in classes.items():
            st.markdown(
                f'<span style="display:inline-block;width:1em;height:1em;background:{color};'
                f'margin-right:0.5em;border:1px solid #000;"></span>{name}',
                unsafe_allow_html=True
            )

# 5. Основная область: загрузка и инференс
st.title("🛰️ Сегментация спутниковых снимков")

uploaded = st.file_uploader("Выберите снимок (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded:
    # Показать исходник
    img = Image.open(uploaded)
    col1, col2 = st.columns([1,1])
    
    col1.subheader("Исходное изображение")
    # Создаем подколонки внутри col1 для уменьшения размера изображения до 1/2
    img_col1, _ = col1.columns(2)
    img_col1.image(img, use_column_width=True)

    # Кнопка сегментации
    if st.button("▶️ Сегментировать"):
        with st.spinner("Идёт сегментация..."):
            try:
                result = call_segmentation_api(uploaded.getvalue(), patch_size, subdivisions)
                # Показать результат
                col2.subheader("Результат сегментации")
                # Создаем подколонки внутри col2 для уменьшения размера изображения до 1/2
                img_col2, _ = col2.columns(2)
                img_col2.image(result, use_column_width=True)
                
                # Ссылка на скачивание
                link = get_image_download_link(result, "segmentation.png", "⬇️ Скачать результат")
                col2.markdown(link, unsafe_allow_html=True)
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка связи с сервером: {e}")
            except Exception as e:
                st.error(f"Неожиданная ошибка: {e}")
else:
    st.info("Загрузите изображение, чтобы начать сегментацию")

# 6. Раздел «О модели»
with st.expander("❓ О модели"):
    st.markdown("""
    ### Описание приложения

    Это приложение выполняет **семантическую сегментацию** спутниковых снимков.  

    Модель обучена различать:
    - **Здания**  
    - **Землю**  
    - **Дороги**  
    - **Растительность**  
    - **Воду**  
    - **Неразмеченные области**  

    ---

    ### Как воспользоваться

    1. **Загрузите снимок**  
    2. При необходимости **скорректируйте параметры** в сайдбаре  
    3. Нажмите **«Сегментировать»**  
    4. **Скачайте маску результата**

    """)

