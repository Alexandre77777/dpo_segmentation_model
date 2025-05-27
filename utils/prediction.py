# utils/prediction.py
import numpy as np
from utils.window import get_window, create_variants, merge_variants

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Предсказание полноразмерной маски с плавными переходами без краевых эффектов
    
    Оригинальный код: https://github.com/Vooban/Smoothly-Blend-Image-Patches
    MIT License, Copyright (c) 2017 Vooban Inc. (Guillaume Chevalier)
    Оптимизировано для ускорения и уменьшения потребления памяти
    """
    # Получаем окно
    window = get_window(window_size)
    
    # Расчёт параметров дополнения и шага
    pad = int(round(window_size * (1 - 1.0/subdivisions)))
    step = window_size // subdivisions
    
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
