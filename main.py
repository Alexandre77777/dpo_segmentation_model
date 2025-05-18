import io
import cv2
import numpy as np
import uvicorn
import os
import gc
import traceback
import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import StreamingResponse
import tensorflow as tf
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler

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

def get_available_memory_mb():
    """
    Определяет доступную системную память в МБ
    Возвращает консервативную оценку доступной памяти
    """
    try:
        # Пробуем использовать psutil для точного определения
        if 'psutil' in globals():
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            return max(50, min(available_mb * 0.8, 400))  # Ограничиваем для безопасности
    except:
        pass
        
    try:
        # Пробуем через /proc/meminfo (для Linux)
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    # Преобразуем кБ в МБ и используем 80% от доступной памяти
                    available = int(line.split()[1]) / 1024 * 0.8
                    return max(50, min(available, 400))  # Не меньше 50МБ, не больше 400МБ
    except:
        pass
    
    # Если не удалось определить, возвращаем консервативное значение
    print("Не удалось определить доступную память, используем значение по умолчанию")
    return 200  # Консервативная оценка для render.com с 512МБ ОЗУ


def memory_optimization_level():
    """
    Определяет уровень оптимизации памяти на основе доступной памяти:
    0 - Нормальный режим (>300 МБ)
    1 - Экономичный режим (200-300 МБ)
    2 - Очень экономичный режим (100-200 МБ)
    3 - Экстремальный режим (<100 МБ)
    """
    available_mb = get_available_memory_mb()
    if available_mb > 300:
        return 0
    elif available_mb > 200:
        return 1
    elif available_mb > 100:
        return 2
    else:
        return 3


def load_model(model_path):
    """Загрузка модели с учетом доступной памяти"""
    global models
    
    if model_path in models:
        return models[model_path]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    # Принудительно очищаем память перед загрузкой модели
    gc.collect()
    mem_level = memory_optimization_level()
    
    # Определяем тип модели по расширению файла
    if model_path.endswith('.tflite'):
        # Настраиваем количество потоков в зависимости от доступной памяти
        num_threads = 1 if mem_level >= 2 else 2
        
        try:
            # Загружаем TFLite модель без экспериментальных делегатов
            interpreter = tf.lite.Interpreter(
                model_path=model_path, 
                num_threads=num_threads
            )
            
            # Выделяем память для тензоров
            interpreter.allocate_tensors()
            
            # Получаем информацию о входных тензорах для диагностики
            input_details = interpreter.get_input_details()
            print(f"TFLite модель загружена. Входные тензоры: {input_details}")
            
            models[model_path] = interpreter
            return interpreter
            
        except Exception as e:
            print(f"Ошибка при загрузке TFLite модели: {str(e)}")
            raise
    else:  # .h5, .keras, и т.д.
        # Отключаем GPU, если память ограничена
        if mem_level >= 2:
            tf.config.set_visible_devices([], 'GPU')
        
        # Ограничиваем число потоков TensorFlow
        if mem_level >= 1:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        
        # Загружаем TensorFlow модель
        model = tf.keras.models.load_model(model_path, compile=False)
        models[model_path] = model
        return model


def is_tflite_model(model):
    """Проверяет, является ли модель TFLite интерпретатором"""
    return isinstance(model, tf.lite.Interpreter)


def predict_with_model(model, image_batch):
    """Предсказание с учетом оптимизации памяти"""
    # Определяем уровень оптимизации
    mem_level = memory_optimization_level()
    
    # Выбираем тип данных в зависимости от уровня оптимизации
    dtype = np.float16 if mem_level >= 1 else np.float32
    
    if is_tflite_model(model):
        try:
            # Получаем информацию о тензорах
            input_details = model.get_input_details()[0]
            output_details = model.get_output_details()[0]
            input_shape = input_details['shape']
            output_shape = output_details['shape']
            
            # Проверяем совместимость размеров
            if len(input_shape) == 4 and input_shape[1] > 1:
                # Если модель требует фиксированный размер входа
                if input_shape[1] != image_batch[0].shape[0] or input_shape[2] != image_batch[0].shape[1]:
                    raise ValueError(
                        f"Размер входного патча {image_batch[0].shape[:2]} не соответствует "
                        f"требуемому размеру модели {input_shape[1:3]}"
                    )
            
            # Управление размером батча в зависимости от доступной памяти
            batch_size = len(image_batch)
            max_batch = 4 if mem_level == 0 else (2 if mem_level == 1 else 1)
            
            # Если батч больше максимального, разбиваем на части
            if batch_size > max_batch:
                results = np.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]), dtype=dtype)
                
                # Обрабатываем небольшими пакетами
                for i in range(0, batch_size, max_batch):
                    end_idx = min(i + max_batch, batch_size)
                    mini_batch = image_batch[i:end_idx]
                    mini_results = predict_with_model(model, mini_batch)
                    results[i:end_idx] = mini_results
                    
                    # Очистка памяти после каждого мини-батча
                    if mem_level >= 2:
                        gc.collect()
                        
                return results
            
            # Для небольших батчей - стандартная обработка
            results = np.zeros((batch_size, output_shape[1], output_shape[2], output_shape[3]), dtype=dtype)
            
            for i, img in enumerate(image_batch):
                # Динамически изменяем размер входного тензора, если нужно
                if not np.array_equal(input_shape, [1, img.shape[0], img.shape[1], img.shape[2]]):
                    model.resize_tensor_input(
                        input_details['index'], 
                        [1, img.shape[0], img.shape[1], img.shape[2]]
                    )
                    model.allocate_tensors()
                    # Обновляем детали после изменения размера
                    input_details = model.get_input_details()[0]
                
                # Устанавливаем входные данные и делаем предсказание
                input_tensor = np.expand_dims(img, axis=0).astype(np.float32)
                model.set_tensor(input_details['index'], input_tensor)
                model.invoke()
                results[i] = model.get_tensor(output_details['index'])[0].astype(dtype)
            
            return results
            
        except Exception as e:
            print(f"Ошибка при предсказании с TFLite: {str(e)}")
            # Пробуем использовать альтернативную модель, если есть
            if 'best_model_float16.h5' in models and model != models['best_model_float16.h5']:
                print("Пробуем использовать H5 модель для предсказания")
                return predict_with_model(models['best_model_float16.h5'], image_batch)
            raise
    else:
        # Для обычной TensorFlow модели
        batch_size = len(image_batch)
        max_batch = 8 if mem_level == 0 else (4 if mem_level == 1 else 2)
        
        # Разбиваем на мини-батчи при необходимости
        if batch_size > max_batch:
            if mem_level >= 2:
                # В режиме сильной оптимизации - обрабатываем по одному
                results = []
                for img in image_batch:
                    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
                    results.append(pred[0])
                    if mem_level >= 3:
                        gc.collect()  # Очищаем память после каждого предсказания
                return np.array(results, dtype=dtype)
            else:
                # В обычном режиме - обрабатываем мини-батчами
                results = np.zeros((batch_size, image_batch[0].shape[0], image_batch[0].shape[1], 6), dtype=dtype)
                
                for i in range(0, batch_size, max_batch):
                    end_idx = min(i + max_batch, batch_size)
                    mini_batch = image_batch[i:end_idx]
                    mini_results = model.predict(mini_batch, verbose=0)
                    results[i:end_idx] = mini_results.astype(dtype)
                    
                    if mem_level >= 1:
                        gc.collect()  # Очищаем память после каждого мини-батча
                        
                return results
        else:
            # Для небольших батчей - стандартная обработка
            results = model.predict(image_batch, verbose=0)
            return results.astype(dtype)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, model):
    """Предсказание с плавными переходами, оптимизированное по памяти"""
    # Определяем уровень оптимизации памяти
    mem_level = memory_optimization_level()
    print(f"Текущий уровень оптимизации памяти: {mem_level}")
    
    # Адаптируем параметры к доступной памяти
    if mem_level >= 1:
        # Уменьшаем размер окна при нехватке памяти
        orig_window_size = window_size
        window_size = min(window_size, 192 if mem_level == 1 else 128)
        if orig_window_size != window_size:
            print(f"Размер окна уменьшен до {window_size} из-за ограничений памяти")
        
        # Уменьшаем значение subdivisions при сильном недостатке памяти
        if mem_level >= 2:
            orig_subdivisions = subdivisions
            subdivisions = 1
            if orig_subdivisions != subdivisions:
                print(f"Subdivisions уменьшено до {subdivisions} из-за сильных ограничений памяти")
    
    # Проверяем совместимость TFLite модели с размером окна
    if is_tflite_model(model):
        input_details = model.get_input_details()[0]
        input_shape = input_details['shape']
        
        # Если TFLite модель требует конкретный размер входа
        if len(input_shape) == 4 and input_shape[1] > 1:
            required_size = input_shape[1]
            if required_size != window_size:
                print(f"TFLite модель требует размер окна {required_size}, а указан {window_size}")
                window_size = required_size
    
    # Расчёт параметров дополнения и шага
    pad = int(round(window_size * (1 - 1.0/subdivisions)))
    step = window_size // subdivisions
    
    # Выбор типа данных в зависимости от уровня оптимизации
    dtype = np.float16 if mem_level >= 1 else np.float32
    
    # Дополняем изображение отражением по краям
    padded = np.pad(input_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    h, w = padded.shape[:2]
    
    # Создаём массив для результата и счётчик наложений
    result = np.zeros((h, w, nb_classes), dtype=dtype)
    counts = np.zeros((h, w, 1), dtype=dtype)
    
    # Создаём весовую функцию, оптимизированную для текущего уровня памяти
    if mem_level >= 2:
        # Простая линейная функция для экономии памяти
        y, x = np.mgrid[0:window_size, 0:window_size]
        center = window_size // 2
        dist_from_center = np.sqrt(((y - center) / center) ** 2 + ((x - center) / center) ** 2)
        window = np.clip(1 - dist_from_center, 0, 1)[:, :, np.newaxis].astype(dtype)
    else:
        # Более качественная функция Блэкмана
        window_1d = np.blackman(window_size)
        window = np.outer(window_1d, window_1d)[:, :, np.newaxis].astype(dtype)
    
    # Определяем оптимальный размер батча для текущей памяти
    batch_size = 8
    if mem_level >= 1:
        batch_size = 4
    if mem_level >= 2:
        batch_size = 2
    if mem_level >= 3:
        batch_size = 1
    
    print(f"Используем батч размером {batch_size} и окно {window_size} для обработки изображения")
    
    # Собираем патчи для пакетного предсказания
    patches = []
    coords = []
    
    # Итерация по патчам с шагом
    for y_start in range(0, h - window_size + 1, step):
        for x_start in range(0, w - window_size + 1, step):
            patch = padded[y_start:y_start+window_size, x_start:x_start+window_size]
            patches.append(patch)
            coords.append((y_start, x_start))
            
            # Если накопили достаточно патчей или достигли конца,
            # выполняем предсказание и очищаем накопленные данные
            if len(patches) >= batch_size or (
                y_start + step >= h - window_size + 1 and x_start + step >= w - window_size + 1
            ):
                if patches:  # Проверяем, что есть патчи для обработки
                    process_batch(patches, coords, window, result, counts, model)
                    patches = []
                    coords = []
                    
                    # Очищаем память в режимах сильной оптимизации
                    if mem_level >= 2:
                        gc.collect()
    
    # Нормализуем по количеству наложений
    mask = counts > 0
    result = np.divide(result, counts + 1e-8, out=result, where=mask)
    
    # Очищаем память
    del counts, mask, window
    gc.collect()
    
    # Обрезаем до исходного размера
    return result[pad:-pad, pad:-pad]


def process_batch(patches, coords, window, result, counts, model):
    """Обработка батча патчей для экономии памяти"""
    # Преобразуем список патчей в массив
    batch_patches = np.array(patches)
    
    try:
        # Получаем предсказания
        predictions = predict_with_model(model, batch_patches)
        
        # Накладываем результаты с весами
        for j, (y_pos, x_pos) in enumerate(coords):
            weighted_pred = predictions[j] * window
            result[y_pos:y_pos+window.shape[0], x_pos:x_pos+window.shape[1]] += weighted_pred
            counts[y_pos:y_pos+window.shape[0], x_pos:x_pos+window.shape[1]] += window
    except Exception as e:
        print(f"Ошибка при обработке батча: {str(e)}")
        # В случае ошибки пробуем обработать патчи по одному
        for i, (patch, (y_pos, x_pos)) in enumerate(zip(patches, coords)):
            try:
                patch_array = np.expand_dims(patch, axis=0)
                pred = predict_with_model(model, patch_array)[0]
                weighted_pred = pred * window
                result[y_pos:y_pos+window.shape[0], x_pos:x_pos+window.shape[1]] += weighted_pred
                counts[y_pos:y_pos+window.shape[0], x_pos:x_pos+window.shape[1]] += window
            except Exception as inner_e:
                print(f"Не удалось обработать патч {i}: {str(inner_e)}")


def label_to_rgb(predicted_image):
    """Преобразование меток классов в RGB изображение"""
    # Цвета для сегментации
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4)))
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4)))
    
    Road = '#6EC1E4'.lstrip('#')
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4)))
    
    Vegetation = 'FEDD3A'.lstrip('#')
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4)))
    
    Water = 'E2A929'.lstrip('#')
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4)))
    
    Unlabeled = '#9B9B9B'.lstrip('#')
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4)))
    
    # Оптимизация: предварительно выделяем память для результата
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)
    
    # Заполняем изображение по классам
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    return segmented_img


@app.on_event("startup")
async def startup_event():
    """Загружаем модели при запуске сервера"""
    # Принудительная очистка памяти перед загрузкой
    gc.collect()
    
    try:
        # Сначала пробуем загрузить TFLite модель как более экономичную
        if os.path.exists('best_model.tflite'):
            load_model('best_model.tflite')
            print("TFLite модель загружена успешно")
        else:
            print("TFLite модель не найдена")
    except Exception as e:
        print(f"Ошибка загрузки TFLite модели: {str(e)}")
    
    try:
        # Затем загружаем H5 модель как запасной вариант
        if os.path.exists('best_model_float16.h5'):
            load_model('best_model_float16.h5')
            print("TensorFlow модель загружена успешно")
        else:
            print("TensorFlow H5 модель не найдена")
    except Exception as e:
        print(f"Ошибка загрузки TensorFlow модели: {str(e)}")
    
    # Выводим информацию о доступной памяти
    available_memory = get_available_memory_mb()
    print(f"Доступная память: {available_memory} МБ")
    print(f"Уровень оптимизации: {memory_optimization_level()}")


@app.get("/")
def read_root():
    """Корневой эндпоинт для проверки работоспособности"""
    mem = get_available_memory_mb()
    return {
        "сообщение": "API модели сегментации работает", 
        "доступная_память_мб": mem,
        "уровень_оптимизации": memory_optimization_level()
    }


@app.get("/models")
def list_available_models():
    """Получение списка доступных моделей"""
    return {"available_models": list(models.keys())}


@app.post("/predict/")
async def predict(
    file: UploadFile = File(...), 
    patch_size: int = 256, 
    subdivisions: int = 2, 
    model_path: str = None,
    max_size: int = 1024  # Максимальный размер стороны изображения
):
    """Эндпоинт для предсказания маски по изображению с оптимизацией памяти"""
    try:
        # Очищаем память перед обработкой нового запроса
        gc.collect()
        
        # Определяем текущий уровень оптимизации
        mem_level = memory_optimization_level()
        available_mb = get_available_memory_mb()
        print(f"Доступная память: {available_mb} МБ, уровень оптимизации: {mem_level}")
        
        # Выбираем оптимальную модель, если не указана
        if model_path is None:
            # В режимах жесткой оптимизации всегда используем TFLite
            if mem_level >= 2 and 'best_model.tflite' in models:
                model_path = 'best_model.tflite'
            # Иначе предпочитаем H5 для более высокого качества
            elif 'best_model_float16.h5' in models:
                model_path = 'best_model_float16.h5'
            # Запасной вариант - любая доступная модель
            elif models:
                model_path = list(models.keys())[0]
            else:
                return JSONResponse(
                    status_code=400, 
                    content={"detail": "Ни одной модели не загружено"}
                )
        
        # Адаптация размера патча и subdivisions к доступной памяти
        if mem_level >= 1:
            # Уменьшаем размер патча при нехватке памяти
            patch_size = min(patch_size, 192 if mem_level == 1 else 128)
            
        if mem_level >= 2:
            # Уменьшаем subdivisions при сильной нехватке памяти
            subdivisions = 1
            
            # Уменьшаем максимальный размер изображения
            max_size = min(max_size, 768 if mem_level == 2 else 512)
        
        # Чтение изображения с минимальным использованием памяти
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Освобождаем память от содержимого файла
        del contents, nparr
        gc.collect()
        
        # Проверяем размер изображения и при необходимости уменьшаем
        h, w = img.shape[:2]
        img_size_mb = (h * w * img.shape[2] * 4) / (1024 * 1024)  # Примерный размер в МБ (float32)
        
        # Адаптивное масштабирование изображения
        if max(h, w) > max_size or img_size_mb > available_mb / 3:
            scale_factor = min(max_size / max(h, w), np.sqrt(available_mb / (3 * img_size_mb)))
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            print(f"Уменьшаем изображение с {w}x{h} до {new_w}x{new_h} из-за ограничений памяти")
            img = cv2.resize(img, (new_w, new_h))
            h, w = new_h, new_w
        
        # Нормализация изображения (выбор оптимального типа данных)
        dtype = np.float16 if mem_level >= 2 else np.float32
        input_img = (img / 255.0).astype(dtype)
        
        # Освобождаем память от исходного изображения
        del img
        gc.collect()
        
        # Загружаем нужную модель
        try:
            model = load_model(model_path)
            print(f"Модель {model_path} загружена успешно")
        except Exception as e:
            print(f"Ошибка загрузки модели {model_path}: {str(e)}")
            # Пробуем найти альтернативную модель
            alternative_found = False
            for alternative in ['best_model_float16.h5', 'best_model.tflite']:
                if alternative != model_path and alternative in models:
                    model_path = alternative
                    model = models[alternative]
                    print(f"Используем альтернативную модель: {model_path}")
                    alternative_found = True
                    break
            
            if not alternative_found:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Не удалось загрузить модель: {str(e)}"}
                )
        
        # Количество классов для сегментации
        n_classes = 6
        
        print(f"Начинаем предсказание с окном {patch_size}px и subdivisions={subdivisions}")
        
        # Предсказание с плавными переходами
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=patch_size,
            subdivisions=subdivisions,
            nb_classes=n_classes,
            model=model
        )
        
        print("Предсказание получено, выполняем argmax")
        
        # Определяем оптимальный способ вычисления argmax
        if mem_level >= 3:
            # Для экстремального режима - обработка по строкам
            final_prediction = np.zeros((h, w), dtype=np.uint8)
            chunk_size = 32  # Небольшой размер чанка для сильной экономии памяти
            
            for i in range(0, h, chunk_size):
                end_i = min(i + chunk_size, h)
                final_prediction[i:end_i] = np.argmax(predictions_smooth[i:end_i], axis=2)
        else:
            # Стандартный подход для нормальной памяти
            final_prediction = np.argmax(predictions_smooth, axis=2)
        
        # Освобождаем память от предсказаний
        del predictions_smooth
        gc.collect()
        
        print("Преобразуем в RGB")
        
        # Преобразование в RGB
        prediction_rgb = label_to_rgb(final_prediction)
        
        # Освобождаем память от предсказания
        del final_prediction
        gc.collect()
        
        print("Готовим изображение для отправки")
        
        # Конвертация в изображение и затем в байты
        result_img = Image.fromarray(prediction_rgb)
        img_byte_arr = io.BytesIO()
        result_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Освобождаем память
        del prediction_rgb, result_img
        gc.collect()
        
        print("Предсказание выполнено успешно")
        return StreamingResponse(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        print(f"Ошибка в обработке: {str(e)}")
        print(traceback.format_exc())
        
        # Аварийная очистка памяти
        gc.collect()
        
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")