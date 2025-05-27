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

# Устанавливаем переменную окружения
os.environ['SM_FRAMEWORK'] = 'tf.keras'

# Импортируем segmentation_models
import segmentation_models as sm

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

# Задаем архитектуру сети и получаем функцию предобработки
BACKBONE = 'efficientnetb1'
preprocess_input = sm.get_preprocessing(BACKBONE)

# Глобальные переменные для хранения модели
model = None
model_type = None  # "tf" или "tflite"
predictor_function = None

# Глобальные переменные для TFLite - инициализируются один раз
_interpreter = None
_input_det = None
_output_det = None
_scale_in = None
_zp_in = None
_scale_out = None
_zp_out = None
_input_shape = None

def load_model_generic(path):
    """
    Загружает .h5/.keras, SavedModel или .tflite модель.
    Возвращает (model, model_type, size_mb).
    """
    global _interpreter, _input_det, _output_det, _scale_in, _zp_in, _scale_out, _zp_out, _input_shape
    
    if path.endswith('.tflite'):
        # Загрузка TFLite модели
        _interpreter = tf.lite.Interpreter(model_path=path)
        _interpreter.allocate_tensors()
        
        _input_det = _interpreter.get_input_details()[0]
        _output_det = _interpreter.get_output_details()[0]
        
        _scale_in, _zp_in = _input_det["quantization"]
        _scale_out, _zp_out = _output_det["quantization"]
        _input_shape = _input_det["shape"]  # [1, H, W, C]
        
        sz = os.path.getsize(path)
        return _interpreter, "tflite", sz / (1024**2)
    
    elif path.endswith(('.h5', '.keras')):
        m = tf.keras.models.load_model(path, compile=False)
        sz = os.path.getsize(path)
        return m, "tf", sz / (1024**2)
    
    else:
        # Предполагаем, что это SavedModel
        m = tf.saved_model.load(path)
        sz = sum(os.path.getsize(os.path.join(r, f))
                for r, _, files in os.walk(path) for f in files)
        return m, "tf", sz / (1024**2)

def create_predictor(model, model_type):
    """
    Создает функцию для предсказания в зависимости от типа модели.
    """
    if model_type == "tflite":
        # Используем точно такую же функцию, как в вашем блокноте
        def predict_fn(patches):
            """
            patches: H×W×C или B×H×W×C (float32)
            возвращает: B×H×W×n_classes (float32)
            """
            global _interpreter, _input_det, _output_det, _scale_in, _zp_in, _scale_out, _zp_out, _input_shape
            
            x = np.asarray(patches, dtype=np.float32)
            if x.ndim == 3:
                x = x[None, ...]  # 1×H×W×C

            # Препроцесс и нормировка
            x = preprocess_input(x) / 255.0

            # Квантование входа (если требуется)
            if _scale_in:
                x = (x / _scale_in + _zp_in).astype(_input_det["dtype"])
            else:
                x = x.astype(_input_det["dtype"])

            # Подгоняем batch_size и инференсим
            batch = x.shape[0]
            _interpreter.resize_tensor_input(
                _input_det["index"], [batch] + list(_input_shape[1:])
            )
            _interpreter.allocate_tensors()
            _interpreter.set_tensor(_input_det["index"], x)
            _interpreter.invoke()

            # Считываем и деквантуем выход
            y = _interpreter.get_tensor(_output_det["index"]).astype(np.float32)
            if _scale_out:
                y = (y - _zp_out) * _scale_out

            return y  # (batch, H, W, n_classes)
            
        return predict_fn
    
    elif model_type == "tf":
        # Для TensorFlow моделей
        if hasattr(model, 'predict'):  # Keras model
            return lambda x: model.predict(preprocess_input(x.astype('float32'))/255., verbose=0)
        else:  # SavedModel
            sig = model.signatures["serving_default"]
            inp_name = list(sig.structured_input_signature[1].keys())[0]
            out_key = list(sig.structured_outputs.keys())[0]
            def predict_fn(x):
                x_prep = preprocess_input(x.astype('float32'))/255.
                t = tf.constant(x_prep)
                return sig(**{inp_name: t})[out_key].numpy()
            return predict_fn

def load_model(model_path='model_f16.tflite'):
    """Загрузка модели из указанного пути"""
    global model, model_type, predictor_function
    if model is None:
        model, model_type, size_mb = load_model_generic(model_path)
        predictor_function = create_predictor(model, model_type)
        print(f"Загружена модель типа {model_type}, размер: {size_mb:.2f} МБ")
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
    # RGB цвета для всех классов
    colors = np.array([
        [60, 16, 152],    # Здание (#3C1098)
        [132, 41, 246],   # Земля (#8429F6)
        [110, 193, 228],  # Дорога (#6EC1E4)
        [254, 221, 58],   # Растительность (#FEDD3A)
        [226, 169, 41],   # Вода (#E2A929)
        [155, 155, 155]   # Неразмеченный (#9B9B9B)
    ], dtype=np.uint8)

    rgb_image = colors[predicted_image]

    return rgb_image

@app.on_event("startup")
async def startup_event():
    """Загружаем модель при запуске сервера"""
    load_model()
    print(f"Модель успешно загружена (тип: {model_type})")

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
        
        # Преобразуем в RGB, как в блокноте
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получение модели и предиктора
        _, predictor = load_model()
        
        # Количество классов
        n_classes = 6
        
        # Предсказание с плавными переходами
        predictions_smooth = predict_img_with_smooth_windowing(
            img,
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
