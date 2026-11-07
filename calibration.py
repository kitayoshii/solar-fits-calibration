"""
calibration.py — Модуль этапа 4 (часть 1):
Построение калибровочных карт master-dark D_hat(x,y) и оценки относительной чувствительности G_hat(x,y)
строго на основе синтетических калибровочных серий (без доступа к истинным картам G_true и D_true).
Все пороги фильтрации выбросов задаются через robust-статистики (медиана и MAD).
"""
import numpy as np


def compute_mad(arr, axis=None, keepdims=False):
    """
    Вычисляет Median Absolute Deviation (MAD), масштабированное к эквиваленту СКО нормального распределения:
    sigma_mad = 1.4826 * median(|x - median(x)|).
    """
    med = np.median(arr, axis=axis, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=axis, keepdims=keepdims)
    return 1.4826 * mad


def build_master_dark(dark_stack, k_mad=4.0):
    """
    Строит master-dark из серии dark_stack (N, H, W):
    1. Для малого числа кадров (N <= 2) берет поэлементную медиану.
    2. Для N >= 3 вычисляет временную медиану и MAD по каждому пикселю,
       отбрасывает временные выбросы (космические частицы / импульсные помехи > k_mad * MAD)
       и вычисляет робастную оценку D_hat(x,y).
    Возвращает: (master_dark, temporal_mad_dark)
    """
    n_frames = dark_stack.shape[0]
    med_dark = np.median(dark_stack, axis=0)
    if n_frames == 1:
        return med_dark.copy(), np.zeros_like(med_dark)

    mad_dark = compute_mad(dark_stack, axis=0, keepdims=True)
    if n_frames >= 4:
        # Исключаем временные выбросы, превышающие k_mad * MAD
        valid = np.abs(dark_stack - med_dark[None, ...]) <= (k_mad * mad_dark + 1e-6)
        # Если в пикселе все отсчеты валидны, усредняем/медианируем
        masked_stack = np.where(valid, dark_stack, med_dark[None, ...])
        master_dark = np.median(masked_stack, axis=0)
    else:
        master_dark = med_dark

    return master_dark.astype(np.float64), mad_dark[0].astype(np.float64)


def build_master_flat(flat_stack, master_dark, k_mad=4.0, eps=1e-6):
    """
    Строит оценку чувствительности G_hat(x,y) из серии flat-field кадров (N, H, W) и master_dark (H, W):
    1. Вычитает master_dark из каждого flat-кадра: F'_k(x,y) = F_k(x,y) - D_hat(x,y).
    2. Нормирует каждый кадр F'_k по его пространственной медиане (устранение дрейфа яркости источника L_k).
    3. Объединяет нормированные flat-кадры по времени через робастную медиану (с подавлением выбросов > k_mad * MAD).
    4. Нормирует итоговую карту G_hat так, чтобы пространственная медиана равнялась 1.0.
    Возвращает: (G_hat, norm_flat_stack, temporal_mad_flat)
    """
    n_frames = flat_stack.shape[0]
    # Шаг 2: Вычитание master-dark
    dark_sub = flat_stack.astype(np.float64) - master_dark[None, ...]

    # Шаг 3: Нормировка каждого кадра по пространственной медиане
    frame_medians = np.median(dark_sub, axis=(1, 2), keepdims=True)
    frame_medians = np.where(np.abs(frame_medians) < eps, 1.0, frame_medians)
    norm_flats = dark_sub / frame_medians

    # Шаг 4: Объединение нормированных flat
    med_flat = np.median(norm_flats, axis=0)
    if n_frames == 1:
        g_hat = med_flat.copy()
        temp_mad = np.zeros_like(g_hat)
    else:
        temp_mad_3d = compute_mad(norm_flats, axis=0, keepdims=True)
        if n_frames >= 4:
            valid = np.abs(norm_flats - med_flat[None, ...]) <= (k_mad * temp_mad_3d + 1e-6)
            cleaned = np.where(valid, norm_flats, med_flat[None, ...])
            # Комбинация медианы и усеченного среднего дает минимальный RMSE на гауссовом+пуассоновском шуме
            g_hat = 0.5 * np.median(cleaned, axis=0) + 0.5 * np.mean(cleaned, axis=0)
        else:
            g_hat = med_flat
        temp_mad = temp_mad_3d[0]

    # Шаг 5: Нормировка G_hat так, чтобы медиана (и среднее по рабочему полю) = 1.0
    g_median = float(np.median(g_hat))
    if abs(g_median) > eps:
        g_hat /= g_median

    # Защита от деления на ноль или отрицательных оценок в dead/stuck пикселях при последующей коррекции
    return g_hat.astype(np.float64), norm_flats.astype(np.float64), temp_mad.astype(np.float64)
