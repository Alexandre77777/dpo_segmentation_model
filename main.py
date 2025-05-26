# main.py
import io
import os
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import StreamingResponse
import tensorflow as tf

# Инициализация FastAPI приложения
app = FastAPI(title="API модели сегментации")

# Настройка CORS для доступа с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для хранения модели
model = None
is_saved_model = False
predictor_function = None

def load_model_generic(path):
    """
    Загружает .h5/.keras или SavedModel.
    Возвращает (model, is_saved_model, size_mb).
    """
    if path.endswith(('.h5', '.keras')):
        m = tf.keras.models.load_model(path, compile=False)
        sz = os.path.getsize(path)
        return m, False, sz / (1024**2)
    m = tf.saved_model.load(path)
    sz = sum(os.path.getsize(os.path.join(r, f))
             for r, _, files in os.walk(path) for f in files)
    return m, True, sz / (1024**2)

def create_predictor(model, is_saved_model):
    """
    Если Keras/H5 — просто model.predict.
    Если SavedModel — берём сигнатуру serving_default.
    Вход — np.ndarray H×W×C, выход — np.ndarray с «логитами» или вероятностями.
    """
    if not is_saved_model:
        return lambda x: model.predict(x.astype('float32')/255., verbose=0)
    sig = model.signatures["serving_default"]
    inp_name = list(sig.structured_input_signature[1].keys())[0]
    out_key = list(sig.structured_outputs.keys())[0]
    def predict_fn(x):
        t = tf.constant(x.astype('float32')/255.)
        return sig(**{inp_name: t})[out_key].numpy()
    return predict_fn

def load_model(model_path='saved_model_dir'):
    """Загрузка модели из указанного пути"""
    global model, is_saved_model, predictor_function
    if model is None:
        model, is_saved_model, _ = load_model_generic(model_path)
        predictor_function = create_predictor(model, is_saved_model)
    return model, predictor_function

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Предсказание полноразмерной маски с плавными переходами без краевых эффектов
    
    Оригинальный код: https://github.com/Vooban/Smoothly-Blend-Image-Patches
    MIT License, Copyright (c) 2017 Vooban Inc. (Guillaume Chevalier)
    Оптимизировано для ускорения и уменьшения потребления памяти
    """
    # Кеширование окон для ускорения
    window_cache = {}
    
    def get_window():
        """Создаёт или возвращает кешированное 2D окно для сглаживания"""
        key = f"{window_size}"
        if key in window_cache:
            return window_cache[key]
            
        # Создаём треугольную функцию
        n = np.arange(1, window_size + 1)
        half_point = (window_size + 1) // 2
        w = np.zeros(window_size)
        w[:half_point] = 2 * n[:half_point] / (window_size + 1)
        w[half_point:] = 2 - 2 * n[half_point:] / (window_size + 1)
        
        # Создаём сплайновую оконную функцию
        intersection = window_size // 4
        wind_outer = (abs(2*w)**2) / 2
        wind_outer[intersection:-intersection] = 0
        
        wind_inner = 1 - (abs(2*(w-1))**2) / 2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0
        
        # Комбинируем и нормализуем
        wind = (wind_inner + wind_outer) / np.mean(wind_inner + wind_outer)
        
        # Создаём 2D окно через внешнее произведение
        window_2d = wind.reshape(-1, 1) @ wind.reshape(1, -1)
        window_2d = window_2d[:, :, np.newaxis]
        
        window_cache[key] = window_2d
        return window_2d
    
    # Получаем окно
    window = get_window()
    
    # Расчёт параметров дополнения и шага
    pad = int(round(window_size * (1 - 1.0/subdivisions)))
    step = window_size // subdivisions
    
    # Функции для создания поворотов и отражений
    def create_variants(img):
        """Создаёт 8 вариантов изображения (повороты и отражения)"""
        variants = []
        # Добавляем оригинал и повороты
        variants.append(img)
        variants.append(np.rot90(img, k=1, axes=(0, 1)))
        variants.append(np.rot90(img, k=2, axes=(0, 1)))
        variants.append(np.rot90(img, k=3, axes=(0, 1)))
        # Добавляем отражение и его повороты
        img_flipped = img[:, ::-1].copy()
        variants.append(img_flipped)
        variants.append(np.rot90(img_flipped, k=1, axes=(0, 1)))
        variants.append(np.rot90(img_flipped, k=2, axes=(0, 1)))
        variants.append(np.rot90(img_flipped, k=3, axes=(0, 1)))
        return variants
    
    def merge_variants(variants):
        """Объединяет результаты 8 вариантов, возвращая их в исходное положение"""
        merged = []
        merged.append(variants[0])
        merged.append(np.rot90(variants[1], k=3, axes=(0, 1)))
        merged.append(np.rot90(variants[2], k=2, axes=(0, 1)))
        merged.append(np.rot90(variants[3], k=1, axes=(0, 1)))
        merged.append(variants[4][:, ::-1])
        merged.append(np.rot90(variants[5], k=3, axes=(0, 1))[:, ::-1])
        merged.append(np.rot90(variants[6], k=2, axes=(0, 1))[:, ::-1])
        merged.append(np.rot90(variants[7], k=1, axes=(0, 1))[:, ::-1])
        return np.mean(merged, axis=0)
    
    # Дополняем изображение отражением по краям
    padded = np.pad(input_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Создаём все варианты изображения (8 вариантов с поворотами/отражениями)
    padded_variants = create_variants(padded)
    
    # Обрабатываем каждый вариант
    results = []
    for variant in padded_variants:
        h, w = variant.shape[:2]
        
        # Создаём массив для результата и счётчик наложений
        result = np.zeros((h, w, nb_classes), dtype=np.float32)
        counts = np.zeros((h, w, 1), dtype=np.float32)
        
        # Собираем патчи для пакетного предсказания
        patches = []
        coords = []
        
        # Итерация по патчам с шагом
        for y in range(0, h - window_size + 1, step):
            for x in range(0, w - window_size + 1, step):
                patch = variant[y:y+window_size, x:x+window_size]
                patches.append(patch)
                coords.append((y, x))
        
        # Пакетное предсказание для всех патчей
        patches_array = np.array(patches)
        predictions = pred_func(patches_array)
        
        # Применяем окно к каждому предсказанию и накладываем в результат
        for idx, (y, x) in enumerate(coords):
            weighted_pred = predictions[idx] * window
            result[y:y+window_size, x:x+window_size] += weighted_pred
            counts[y:y+window_size, x:x+window_size] += window
        
        # Нормализуем по количеству наложений
        result = np.divide(result, counts + 1e-8, out=result, where=counts > 0)
        
        # Обрезаем до исходного размера (без дополнения)
        results.append(result[pad:-pad, pad:-pad])
    
    # Объединяем все варианты и устраняем повороты/отражения
    merged_result = merge_variants(results)
    
    # Обрезаем по размеру исходного изображения
    return merged_result[:input_img.shape[0], :input_img.shape[1]]

def label_to_rgb(predicted_image):
    """Преобразование меток классов в RGB изображение"""
    # Здание
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4)))

    # Земля
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4)))

    # Дорога
    Road = '#6EC1E4'.lstrip('#')
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4)))

    # Растительность
    Vegetation = 'FEDD3A'.lstrip('#')
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4)))

    # Вода
    Water = 'E2A929'.lstrip('#')
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4)))

    # Неразмеченный
    Unlabeled = '#9B9B9B'.lstrip('#')
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4)))

    # Создание пустого изображения с тремя каналами (RGB)
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))

    # Заполнение изображения соответствующими цветами в зависимости от меток классов
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled

    # Преобразование типа данных в uint8 для корректного отображения изображения
    segmented_img = segmented_img.astype(np.uint8)

    return segmented_img

@app.on_event("startup")
async def startup_event():
    """Загружаем модель при запуске сервера"""
    load_model()
    print("Модель успешно загружена")

@app.get("/")
def read_root():
    """Корневой эндпоинт для проверки работоспособности"""
    return {"сообщение": "API модели сегментации работает"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...), patch_size: int = 256, subdivisions: int = 2):
    """Эндпоинт для предсказания сегментации по изображению"""
    try:
        # Чтение и обработка изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Получение модели и предиктора
        _, predictor = load_model()
        
        # Количество классов
        n_classes = 6
        
        # Предсказание с плавными переходами
        predictions_smooth = predict_img_with_smooth_windowing(
            img,  # Передаем необработанное изображение, предобработка внутри predictor
            window_size=patch_size,
            subdivisions=subdivisions,
            nb_classes=n_classes,
            pred_func=predictor
        )
        
        # Получение итогового предсказания
        final_prediction = np.argmax(predictions_smooth, axis=2)
        
        # Преобразование в RGB
        prediction_rgb = label_to_rgb(final_prediction)
        
        # Конвертация в изображение и затем в байты
        result_img = Image.fromarray(prediction_rgb)
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки изображения: {str(e)}")