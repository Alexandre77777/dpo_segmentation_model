# utils/model_loader.py
import os
import tensorflow as tf
import numpy as np
import segmentation_models as sm
from config import BACKBONE, DEFAULT_MODEL_PATH, ENV_CONFIG

# Устанавливаем переменные окружения
for key, value in ENV_CONFIG.items():
    os.environ[key] = value

# Получаем функцию предобработки
preprocess_input = sm.get_preprocessing(BACKBONE)

# Глобальные переменные для TFLite - инициализируются один раз
_interpreter = None
_input_det = None
_output_det = None
_scale_in = None
_zp_in = None
_scale_out = None
_zp_out = None
_input_shape = None

# Глобальные переменные для хранения модели
model = None
model_type = None
predictor_function = None

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

def load_model(model_path=DEFAULT_MODEL_PATH):
    """Загрузка модели из указанного пути"""
    global model, model_type, predictor_function
    if model is None:
        model, model_type, size_mb = load_model_generic(model_path)
        predictor_function = create_predictor(model, model_type)
        print(f"Загружена модель типа {model_type}, размер: {size_mb:.2f} МБ")
    return model, predictor_function
