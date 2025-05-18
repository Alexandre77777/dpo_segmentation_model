import io
import cv2
import numpy as np
import uvicorn
import os
import gc
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import StreamingResponse
import tensorflow as tf

# Инициализация FastAPI приложения
app = FastAPI(title="API сегментации (ультра-оптимизация)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальная переменная для модели
model = None

def inspect_tflite_model(model_path):
    """Проверяет структуру TFLite модели и выводит детали"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        print("=== ДЕТАЛИ МОДЕЛИ ===")
        print(f"Входной тензор: {input_details}")
        print(f"Выходной тензор: {output_details}")
        print(f"Форма входа: {input_details['shape']}")
        print(f"Форма выхода: {output_details['shape']}")
        print("=====================")
        
        return interpreter, input_details['shape'], output_details['shape']
    except Exception as e:
        print(f"Ошибка при анализе модели: {str(e)}")
        return None, None, None

def predict_h5_model(img_path):
    """Запасное предсказание с использованием H5 модели"""
    print("Используем H5 модель для предсказания")
    
    try:
        # Загружаем H5 модель
        h5_model = tf.keras.models.load_model('best_model_float16.h5', compile=False)
        
        # Читаем изображение
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Нормализуем и изменяем размер если нужно
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # Нормализуем
        input_img = img.astype(np.float32) / 255.0
        
        # Добавляем размерность батча
        input_img = np.expand_dims(input_img, axis=0)
        
        # Получаем предсказание
        prediction = h5_model.predict(input_img)[0]
        
        # Получаем классы
        final_mask = np.argmax(prediction, axis=-1)
        
        # Освобождаем память
        del h5_model, input_img, prediction
        gc.collect()
        
        return final_mask
    except Exception as e:
        print(f"Ошибка при использовании H5 модели: {str(e)}")
        return None

def ultra_minimal_tflite_inference(interpreter, img_path, max_size=512):
    """
    Экстремально минималистичная функция для инференса TFLite модели
    с минимальным потреблением памяти
    """
    try:
        print("Начало предсказания с ультра-минимальным использованием памяти")
        
        # Получаем детали модели
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        input_shape = input_details['shape']
        output_shape = output_details['shape']
        
        print(f"Модель ожидает входную форму: {input_shape}")
        
        # Проверяем, может ли модель использоваться для сегментации
        if input_shape[1] == 1 or input_shape[2] == 1:
            print("TFLite модель ожидает вход с размерностью 1, используем H5 модель")
            return predict_h5_model(img_path)
        
        # Читаем изображение напрямую из файла для экономии памяти
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Получаем размеры
        orig_h, orig_w = img.shape[:2]
        
        # Проверяем, нужно ли уменьшить изображение
        if max(orig_h, orig_w) > max_size:
            scale = max_size / max(orig_h, orig_w)
            h, w = int(orig_h * scale), int(orig_w * scale)
            img = cv2.resize(img, (w, h))
            print(f"Изображение уменьшено до {w}x{h}")
        else:
            h, w = orig_h, orig_w
        
        # Нормализуем изображение
        img = img.astype(np.float32) / 255.0
        
        # Подготавливаем входной тензор с нужной формой
        if input_shape[0] == 1 and input_shape[1] > 1 and input_shape[2] > 1:
            # Модель ожидает фиксированный размер входа
            model_h, model_w = input_shape[1], input_shape[2]
            
            # Изменяем размер для соответствия модели
            if h != model_h or w != model_w:
                img_resized = cv2.resize(img, (model_w, model_h))
                input_tensor = np.expand_dims(img_resized, axis=0)
                print(f"Изображение изменено до {model_w}x{model_h} для соответствия модели")
            else:
                input_tensor = np.expand_dims(img, axis=0)
                
        # Если модель ожидает динамический размер [1,-1,-1,3]
        elif input_shape[0] == -1 or input_shape[1] == -1 or input_shape[2] == -1:
            # Подготавливаем входной тензор
            input_tensor = np.expand_dims(img, axis=0)
            
            # Изменяем размер входного тензора
            interpreter.resize_tensor_input(input_details['index'], input_tensor.shape)
            interpreter.allocate_tensors()
            print(f"Модель перенастроена под размер {input_tensor.shape}")
        else:
            # Последняя попытка - изменяем до зафиксированного размера
            model_h, model_w = input_shape[1] or 224, input_shape[2] or 224
            img_resized = cv2.resize(img, (model_w, model_h))
            input_tensor = np.expand_dims(img_resized, axis=0)
            print(f"Используем размер по умолчанию {model_w}x{model_h}")
            
        # Освобождаем память
        del img
        gc.collect()
        
        # Выполняем инференс
        print(f"Запускаем инференс с формой {input_tensor.shape}")
        interpreter.set_tensor(input_details['index'], input_tensor)
        interpreter.invoke()
        
        # Получаем результат
        output_tensor = interpreter.get_tensor(output_details['index'])
        print(f"Получен результат с формой {output_tensor.shape}")
        
        # Освобождаем память
        del input_tensor
        gc.collect()
        
        # Возвращаем первый результат и определяем классы
        if len(output_tensor.shape) == 4:
            predictions = output_tensor[0]
            
            # Возвращаем к исходному размеру если нужно
            if predictions.shape[0] != h or predictions.shape[1] != w:
                predictions_resized = cv2.resize(predictions, (w, h), 
                                               interpolation=cv2.INTER_LINEAR)
                final_mask = np.argmax(predictions_resized, axis=-1)
            else:
                final_mask = np.argmax(predictions, axis=-1)
        else:
            # В случае неожиданного формата выхода
            print(f"Необычный формат выхода: {output_tensor.shape}")
            return None
        
        # Возвращаем маску, преобразованную к исходному размеру
        if h != orig_h or w != orig_w:
            final_mask = cv2.resize(final_mask.astype(np.uint8), (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST)
        
        # Освобождаем память
        del output_tensor, predictions
        gc.collect()
        
        return final_mask
        
    except Exception as e:
        print(f"Ошибка при инференсе: {str(e)}")
        print(traceback.format_exc())
        return None

def label_to_rgb(predicted_image):
    """Преобразование маски классов в RGB изображение"""
    colors = [
        [60, 16, 152],    # Building - #3C1098
        [132, 41, 246],   # Land - #8429F6
        [110, 193, 228],  # Road - #6EC1E4
        [254, 221, 58],   # Vegetation - FEDD3A
        [226, 169, 41],   # Water - E2A929
        [155, 155, 155]   # Unlabeled - #9B9B9B
    ]
    
    # Создаем RGB изображение с оптимизированной памятью
    h, w = predicted_image.shape[:2]
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Заполняем по одному классу за раз для экономии памяти
    for class_idx, color in enumerate(colors):
        mask = predicted_image == class_idx
        rgb_mask[mask] = color
    
    return rgb_mask

@app.on_event("startup")
async def startup_event():
    """Проверяем и загружаем модель при запуске"""
    try:
        # Проверяем наличие TFLite модели
        if os.path.exists('best_model.tflite'):
            print("TFLite модель найдена, анализируем...")
            model_data = inspect_tflite_model('best_model.tflite')
            global model
            model = model_data[0]
        else:
            print("TFLite модель не найдена")
    except Exception as e:
        print(f"Ошибка инициализации: {str(e)}")

@app.get("/")
def read_root():
    """Корневой эндпоинт для проверки работоспособности"""
    return {"сообщение": "API сегментации работает"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...), max_size: int = 512):
    """
    Эндпоинт предсказания с ультра-низким использованием памяти
    """
    # Очищаем память перед началом
    gc.collect()
    
    try:
        global model
        
        # Проверяем, загружена ли модель
        if model is None:
            if os.path.exists('best_model.tflite'):
                model, _, _ = inspect_tflite_model('best_model.tflite')
                if model is None:
                    raise HTTPException(status_code=500, detail="Не удалось загрузить модель")
            else:
                raise HTTPException(status_code=404, detail="Модель не найдена")
        
        # Создаем временный файл для сохранения загруженного изображения
        # Это позволяет не хранить все изображение в памяти
        temp_file = f"temp_image_{hash(file.filename)}.jpg"
        try:
            # Сохраняем загруженное изображение во временный файл
            with open(temp_file, "wb") as buffer:
                # Читаем и сохраняем по частям
                chunk_size = 1024 * 64  # 64KB чанки
                while content := await file.read(chunk_size):
                    buffer.write(content)
            
            # Запускаем предсказание с минимальным использованием памяти
            prediction_mask = ultra_minimal_tflite_inference(model, temp_file, max_size)
            
            # Если TFLite не справился, используем H5 (уже внутри функции)
            if prediction_mask is None:
                raise ValueError("TFLite модель не смогла обработать изображение")
            
            # Преобразуем маску в RGB
            rgb_mask = label_to_rgb(prediction_mask)
            
            # Освобождаем память
            del prediction_mask
            gc.collect()
            
            # Преобразуем в PNG
            pil_img = Image.fromarray(rgb_mask)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Освобождаем память
            del rgb_mask, pil_img
            gc.collect()
            
            return StreamingResponse(img_byte_arr, media_type="image/png")
            
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
    except Exception as e:
        gc.collect()
        print(f"Ошибка: {str(e)}")
        print(traceback.format_exc())
        
        # В случае ошибки TFLite пробуем H5
        try:
            # Создаем временный файл для загрузки
            temp_file = f"temp_fallback_{hash(file.filename)}.jpg"
            
            # Перемещаем указатель в начало
            await file.seek(0)
            
            # Сохраняем во временный файл
            with open(temp_file, "wb") as buffer:
                buffer.write(await file.read())
            
            # Используем H5 модель напрямую
            fallback_mask = predict_h5_model(temp_file)
            
            if fallback_mask is not None:
                # Преобразуем в RGB
                fallback_rgb = label_to_rgb(fallback_mask)
                
                # Преобразуем в PNG
                pil_img = Image.fromarray(fallback_rgb)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                # Очищаем
                del fallback_mask, fallback_rgb, pil_img
                gc.collect()
                
                # Удаляем временный файл
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                return StreamingResponse(img_byte_arr, media_type="image/png")
            else:
                # Удаляем временный файл
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
                raise HTTPException(
                    status_code=500, 
                    detail=f"Не удалось выполнить предсказание: {str(e)}"
                )
                
        except Exception as fallback_e:
            # Если и запасной вариант не сработал
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка предсказания (основная: {str(e)}, запасная: {str(fallback_e)})"
            )

