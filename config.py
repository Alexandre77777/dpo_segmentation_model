# config.py

# Настройки модели
BACKBONE = 'efficientnetb1'
DEFAULT_MODEL_PATH = 'model_f16.tflite'
ENV_CONFIG = {'SM_FRAMEWORK': 'tf.keras'}

# Настройки приложения
APP_TITLE = "API модели сегментации"

# Настройки CORS
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Цвета для сегментации
SEGMENTATION_COLORS = [
    [60, 16, 152],    # Здание (#3C1098)
    [132, 41, 246],   # Земля (#8429F6)
    [110, 193, 228],  # Дорога (#6EC1E4)
    [254, 221, 58],   # Растительность (#FEDD3A)
    [226, 169, 41],   # Вода (#E2A929)
    [155, 155, 155]   # Неразмеченный (#9B9B9B)
]

# Параметры по умолчанию
DEFAULT_PATCH_SIZE = 256
DEFAULT_SUBDIVISIONS = 2
DEFAULT_NUM_CLASSES = 6

# Выбор алгоритма предсказания
USE_SIMPLE_ALGORITHM = True  # По умолчанию используем оптимизированную версию
