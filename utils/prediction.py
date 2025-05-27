# utils/prediction.py
import numpy as np
from utils.window import get_window, create_variants, merge_variants

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Предсказание с плавными переходами между патчами
    Поддерживает subdivisions >= 1
    """
    # Получаем окно
    window = get_window(window_size)
    
    # Расчёт параметров дополнения и шага с учётом subdivisions = 1
    if subdivisions == 1:
        # Для случая subdivisions=1 используем минимальное перекрытие 25%
        minimal_overlap = window_size // 4
        pad = minimal_overlap
        step = window_size - minimal_overlap
    else:
        # Стандартный расчёт для subdivisions > 1
        pad = int(round(window_size * (1 - 1.0/subdivisions)))
        step = window_size // subdivisions
    
    # Дополняем изображение отражением по краям
    padded = np.pad(input_img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Создаём все варианты изображения (повороты/отражения)
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
        
        for y in range(0, h - window_size + 1, step):
            for x in range(0, w - window_size + 1, step):
                patch = variant[y:y+window_size, x:x+window_size]
                patches.append(patch)
                coords.append((y, x))
        
        # Пакетное предсказание
        patches_array = np.array(patches)
        predictions = pred_func(patches_array)
        
        # Применяем окно к каждому предсказанию
        for idx, (y, x) in enumerate(coords):
            weighted_pred = predictions[idx] * window
            result[y:y+window_size, x:x+window_size] += weighted_pred
            counts[y:y+window_size, x:x+window_size] += window
        
        # Нормализуем по количеству наложений
        result = np.divide(result, counts + 1e-8, out=result, where=counts > 0)
        results.append(result[pad:-pad, pad:-pad])
    
    # Объединяем все варианты
    merged_result = merge_variants(results)
    return merged_result[:input_img.shape[0], :input_img.shape[1]]

def predict_img_simple_tiling(input_img, window_size, overlap, nb_classes, pred_func):
    """
    Простая и эффективная версия предсказания с минимальным перекрытием.
    Гораздо быстрее и использует меньше памяти, чем predict_img_with_smooth_windowing.
    
    Параметры:
        overlap: размер перекрытия между патчами (в пикселях)
    """
    h, w = input_img.shape[:2]
    
    # Если изображение меньше окна, просто предсказываем целиком
    if h <= window_size and w <= window_size:
        if h < window_size or w < window_size:
            padded = np.zeros((max(h, window_size), max(w, window_size), input_img.shape[2]), dtype=input_img.dtype)
            padded[:h, :w] = input_img
            prediction = pred_func(padded[np.newaxis, ...])[0]
            return prediction[:h, :w]
        else:
            return pred_func(input_img[np.newaxis, ...])[0]
    
    # Разбиваем на патчи с перекрытием
    step = window_size - overlap
    
    # Количество шагов по вертикали и горизонтали
    n_h = 1 + (h - window_size + step - 1) // step
    n_w = 1 + (w - window_size + step - 1) // step
    
    # Массивы для результата и счетчика
    prediction = np.zeros((h, w, nb_classes), dtype=np.float32)
    counter = np.zeros((h, w, 1), dtype=np.float32)
    
    # Создаем маску для плавного смешивания
    if overlap > 0:
        mask_h = np.ones((window_size, 1))
        mask_w = np.ones((1, window_size))
        
        # Линейные рампы для перехода
        ramp = np.linspace(0, 1, overlap)
        mask_h[:overlap, 0] = ramp
        mask_h[-overlap:, 0] = ramp[::-1]
        mask_w[0, :overlap] = ramp
        mask_w[0, -overlap:] = ramp[::-1]
        
        # 2D маска
        mask = mask_h * mask_w
        mask = mask[:, :, np.newaxis]
    else:
        mask = np.ones((window_size, window_size, 1), dtype=np.float32)
    
    # Создаем батчи патчей для эффективного предсказания
    patches = []
    coords = []
    
    for i in range(n_h):
        for j in range(n_w):
            y_start = min(i * step, h - window_size)
            x_start = min(j * step, w - window_size)
            
            # Извлекаем патч
            patch = input_img[y_start:y_start+window_size, x_start:x_start+window_size]
            
            # Дополняем патч, если он меньше окна
            if patch.shape[0] < window_size or patch.shape[1] < window_size:
                temp_patch = np.zeros((window_size, window_size, input_img.shape[2]), dtype=input_img.dtype)
                temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = temp_patch
            
            patches.append(patch)
            coords.append((y_start, x_start))
    
    # Пакетное предсказание
    patches_array = np.array(patches)
    predictions = pred_func(patches_array)
    
    # Встраиваем предсказания обратно в изображение
    for idx, (y_start, x_start) in enumerate(coords):
        weighted_pred = predictions[idx] * mask
        
        h_patch = min(window_size, h - y_start)
        w_patch = min(window_size, w - x_start)
        
        prediction[y_start:y_start+h_patch, x_start:x_start+w_patch] += weighted_pred[:h_patch, :w_patch]
        counter[y_start:y_start+h_patch, x_start:x_start+w_patch] += mask[:h_patch, :w_patch]
    
    # Нормализуем результат
    prediction = np.divide(prediction, counter + 1e-8, out=prediction, where=counter > 0)
    
    return prediction
