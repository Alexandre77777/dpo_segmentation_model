# main.py
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from config import APP_TITLE, CORS_CONFIG, DEFAULT_PATCH_SIZE, DEFAULT_SUBDIVISIONS, DEFAULT_NUM_CLASSES
from utils.model_loader import load_model
from utils.prediction import predict_img_with_smooth_windowing
from utils.image_processing import load_image_from_bytes, label_to_rgb, create_response_image

# Инициализация FastAPI приложения
app = FastAPI(title=APP_TITLE)

# Настройка CORS для доступа с фронтенда
app.add_middleware(
    CORSMiddleware,
    **CORS_CONFIG
)

@app.on_event("startup")
async def startup_event():
    """Загружаем модель при запуске сервера"""
    _, model_type = load_model()
    print(f"Модель успешно загружена (тип: {model_type})")

@app.get("/")
def read_root():
    """Корневой эндпоинт для проверки работоспособности"""
    return {"сообщение": "API модели сегментации работает"}

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), 
    patch_size: int = DEFAULT_PATCH_SIZE, 
    subdivisions: int = DEFAULT_SUBDIVISIONS
):
    """Эндпоинт для предсказания сегментации по изображению"""
    try:
        # Чтение и обработка изображения
        contents = await file.read()
        img = load_image_from_bytes(contents)
        
        # Получение модели и предиктора
        _, predictor = load_model()
        
        # Предсказание с плавными переходами
        predictions_smooth = predict_img_with_smooth_windowing(
            img,
            window_size=patch_size,
            subdivisions=subdivisions,
            nb_classes=DEFAULT_NUM_CLASSES,
            pred_func=predictor
        )
        
        # Получение итогового предсказания
        final_prediction = np.argmax(predictions_smooth, axis=2)
        
        # Преобразование в RGB
        prediction_rgb = label_to_rgb(final_prediction)
        
        # Создание ответа
        img_byte_arr = create_response_image(prediction_rgb)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")
