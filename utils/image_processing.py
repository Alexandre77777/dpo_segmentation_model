# utils/image_processing.py
import cv2
import numpy as np
from PIL import Image
import io
from config import SEGMENTATION_COLORS

def load_image_from_bytes(image_bytes):
    """Загружает изображение из байтов"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB

def label_to_rgb(predicted_image):
    """Преобразует маску классов в RGB изображение"""
    colors = np.array(SEGMENTATION_COLORS, dtype=np.uint8)
    rgb_image = colors[predicted_image]
    return rgb_image

def create_response_image(prediction_rgb):
    """Создает байтовый объект изображения для ответа"""
    result_img = Image.fromarray(prediction_rgb)
    img_byte_arr = io.BytesIO()
    result_img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr
