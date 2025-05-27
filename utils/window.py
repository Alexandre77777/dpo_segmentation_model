# utils/window.py
import numpy as np

# Кеширование окон для ускорения
window_cache = {}

def get_window(window_size):
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
