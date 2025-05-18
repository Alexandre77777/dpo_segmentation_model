# main.py
import io
import cv2
import numpy as np
import uvicorn
import os
import gc
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import StreamingResponse
import tensorflow as tf
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler

# Отключаем все экспериментальные делегаты
os.environ["TFLITE_DISABLE_DELEGATE_CLUSTERING"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# Глобальные переменные для хранения моделей
models = {}

def get_available_memory_mb(default_mb=300):
    """Определяет доступную память в МБ - упрощенная версия без чтения proc"""
    # Возвращаем фиксированное значение для надежности
    return default_mb


def load_model(model_path):
    """Загрузка модели по пути (определяет тип модели по расширению файла)"""
    global models
    
    if model_path in models:
        return models[model_path]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    # Определяем тип модели по расширению файла
    if model_path.endswith('.tflite'):
        try:
            # Загружаем TFLite модель без экспериментальных делегатов
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=1
            )
            interpreter.allocate_tensors()
            models[model_path] = interpreter
            return interpreter
        except Exception as e:
            print(f"Ошибка загрузки TFLite модели: {str(e)}")
            raise
    else:  # .h5, .keras, и т.д.
        # Загружаем TensorFlow модель
        model = tf.keras.models.load_model(model_path, compile=False)
        models[model_path] = model
        return model


def is_tflite_model(model):
    """Проверяет, является ли модель TFLite интерпретатором"""
    return isinstance(model, tf.lite.Interpreter)


def predict_with_model(model, image_batch):
    """Выполнение предсказания с базовыми настройками без оптимизации"""
    if is_tflite_model(model):
        # Получаем информацию о входном/выходном тензорах
        input_details = model.get_input_details()[0]
        output_details = model.get_output_details()[0]
        
        # Получаем форму выхода
        output_shape = output_details['shape']
        
        # Подготавливаем результирующий массив для предсказаний
        batch_size = len(image_batch)
        results = np.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]), dtype=np.float32)
        
        # Обрабатываем каждое изображение в батче отдельно
        for i, img in enumerate(image_batch):
            # Устанавливаем данные входного тензора
            model.set_tensor(input_details['index'], np.expand_dims(img, axis=0).astype(np.float32))
            
            # Выполняем инференс
            model.invoke()
            
            # Получаем выходной тензор
            output_data = model.get_tensor(output_details['index'])
            
            # Сохраняем результат
            results[i] = output_data[0]
        
        return results
    else:
        # Обычная TensorFlow модель
        return model.predict(image_batch, verbose=0)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, model):
    """
    Предсказание с плавными переходами - возвращает к базовой версии, 
    которая раньше работала
    """
    # Расчёт параметров дополнения и шага
    pad = int(round(window_size * (1 - 1.0/subdivisions)))
    step = window_size // subdivisions
    
    # Дополняем изображение отражением по краям
    padded = np.pad(input_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w = padded.shape[:2]
    
    # Создаём массив для результата и счётчик наложений
    result = np.zeros((h, w, nb_classes), dtype=np.float32)
    counts = np.zeros((h, w, 1), dtype=np.float32)
    
    # Создаём весовую функцию для перекрытия окон
    y, x = np.mgrid[0:window_size, 0:window_size]
    center = window_size // 2
    dist_from_center = np.sqrt(((x - center) / center) ** 2 + ((y - center) / center) ** 2)
    window = np.clip(1 - dist_from_center, 0, 1)[:, :, np.newaxis]
    
    # Собираем патчи для пакетного предсказания
    patches = []
    coords = []
    
    # Итерация по патчам с шагом
    for y_start in range(0, h - window_size + 1, step):
        for x_start in range(0, w - window_size + 1, step):
            patch = padded[y_start:y_start+window_size, x_start:x_start+window_size]
            patches.append(patch)
            coords.append((y_start, x_start))
    
    # Разбиваем на более мелкие батчи для обработки
    batch_size = 4  # Используем небольшой размер для надежности
    for i in range(0, len(patches), batch_size):
        batch_patches = np.array(patches[i:i+batch_size])
        batch_coords = coords[i:i+batch_size]
        
        # Пакетное предсказание для текущего батча
        predictions = predict_with_model(model, batch_patches)
        
        # Применяем окно к каждому предсказанию и накладываем в результат
        for j, (y, x) in enumerate(batch_coords):
            weighted_pred = predictions[j] * window
            result[y:y+window_size, x:x+window_size] += weighted_pred
            counts[y:y+window_size, x:x+window_size] += window
    
    # Нормализуем по количеству наложений
    mask = counts > 0
    result = np.divide(result, counts, out=result, where=mask)
    
    # Обрезаем до исходного размера
    return result[pad:-pad, pad:-pad]


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

    # Создание изображения с тремя каналами (RGB)
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


def convert_h5_to_optimized_tflite(h5_model_path, tflite_output_path):
    """Конвертирует H5 модель в оптимизированный TFLite формат с приоритетом на уменьшение размера"""
    # Загружаем модель
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    
    # Создаем конвертер и настраиваем оптимизации
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    
    # Конвертируем и сохраняем модель
    tflite_model = converter.convert()
    with open(tflite_output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Выводим информацию о размерах
    original_size = os.path.getsize(h5_model_path) / (1024*1024)
    tflite_size = os.path.getsize(tflite_output_path) / (1024*1024)
    print(f"Исходный размер: {original_size:.2f} МБ")
    print(f"TFLite размер: {tflite_size:.2f} МБ")
    print(f"Коэффициент сжатия: {original_size/tflite_size:.2f}x")


@app.get("/")
def read_root():
    """Корневой эндпоинт для проверки работоспособности"""
    return {"сообщение": "API модели сегментации работает"}


@app.get("/models")
def list_available_models():
    """Получение списка доступных моделей"""
    return {"available_models": list(models.keys())}


@app.post("/predict/")
async def predict(file: UploadFile = File(...), 
                patch_size: int = 256, 
                subdivisions: int = 2, 
                model_path: str = "best_model_float16.h5"):
    """Эндпоинт для предсказания маски по изображению"""
    try:
        # Добавляем логирование для отладки
        print(f"Получен запрос с patch_size={patch_size}, subdivisions={subdivisions}, model_path={model_path}")
        
        # Чтение и обработка изображения
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        print(f"Изображение загружено, размер: {img.shape}")
        
        # Нормализация изображения
        input_img = img.astype(np.float32) / 255.0
        
        # Загружаем нужную модель
        try:
            model = load_model(model_path)
            print(f"Модель {model_path} загружена успешно")
        except FileNotFoundError as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            # Пробуем запасную модель
            if model_path != "best_model_float16.h5" and os.path.exists("best_model_float16.h5"):
                print("Пробуем запасную модель best_model_float16.h5")
                model = load_model("best_model_float16.h5")
            else:
                print("Запасная модель не найдена")
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Модель не найдена: {model_path}"}
                )
        
        # Ограничиваем размеры входного изображения для безопасности
        if max(img.shape[:2]) > 1024:
            print("Уменьшаем размер изображения до максимум 1024 px для безопасности")
            scale = 1024 / max(img.shape[:2])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            input_img = cv2.resize(input_img, new_size)
        
        # Количество классов
        n_classes = 6
        
        print(f"Начинаем предсказание с окном {patch_size}px и {subdivisions} subdivisions")
        # Предсказание с плавными переходами
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=patch_size,
            subdivisions=subdivisions,
            nb_classes=n_classes,
            model=model
        )
        
        print("Предсказание получено, выполняем argmax")
        # Получение итогового предсказания
        final_prediction = np.argmax(predictions_smooth, axis=2)
        
        print("Преобразуем в RGB")
        # Преобразование в RGB
        prediction_rgb = label_to_rgb(final_prediction)
        
        print("Готовим изображение для отправки")
        # Конвертация в изображение и затем в байты
        result_img = Image.fromarray(prediction_rgb)
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        print("Отправляем ответ")
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        print(f"Ошибка в обработке: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")