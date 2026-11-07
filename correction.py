"""
correction.py — Модуль этапа 5:
Радиометрическая коррекция научного кадра S_hat(x,y) = (Y(x,y) - D_hat(x,y)) / G_hat(x,y)
и восстановление (реконструкция) дефектных пикселей тремя методами:
1. Билинейная/бигармоническая интерполяция по регулярной сетке (griddata / convolve).
2. Локальная медиана соседей (median neighbor filtering).
3. Взвешенная обратно-дистанционная/гауссова интерполяция (weighted IDW/Gaussian convolution).
Сохранение исправленного FITS с записью информации о калибровке в header.
"""
import numpy as np
from scipy.ndimage import median_filter, convolve
from scipy.interpolate import griddata
from fits_utils import write_fits


def radiometric_calibrate(y_frame, master_dark, g_hat, g_min_clip=0.05):
    """
    Выполняет базовую плоскую и темновую коррекцию:
    S_hat(x,y) = (Y(x,y) - D_hat(x,y)) / G_hat(x,y).
    Для предотвращения деления на 0 в dead/stuck пикселях до этапа интерполяции
    значения G_hat ниже g_min_clip ограничиваются снизу.
    """
    g_safe = np.where(np.abs(g_hat) < g_min_clip, g_min_clip, g_hat)
    s_hat_raw = (y_frame.astype(np.float64) - master_dark.astype(np.float64)) / g_safe
    return s_hat_raw


def inpaint_median_neighbors(image, bad_mask, window=3):
    """
    Метод 1: Восстановление дефектных пикселей медианой окрестности.
    Если внутри окна 3x3 все соседи дефектные, подставляется медиана из окна 5x5.
    """
    repaired = image.copy()
    if not np.any(bad_mask):
        return repaired

    med3 = median_filter(repaired, size=window, mode='reflect')
    med5 = median_filter(repaired, size=window + 2, mode='reflect')
    repaired[bad_mask] = med3[bad_mask]
    # Второй проход для кластеров из 2-3 соседних битых пикселей
    repaired_pass2 = median_filter(repaired, size=window, mode='reflect')
    repaired[bad_mask] = 0.7 * repaired_pass2[bad_mask] + 0.3 * med5[bad_mask]
    return repaired


def inpaint_bilinear(image, bad_mask):
    """
    Метод 2: Билинейная интерполяция по исправным пикселям окрестности.
    Для высокой скорости на матрицах 1024x1024 использует точную линейную интерполяцию
    по 4 ближайшим ортогональным соседям с итеративным уточнением (решение уравнения Лапласа на маске)
    или scipy.interpolate.griddata в локальных окнах.
    """
    repaired = image.copy()
    if not np.any(bad_mask):
        return repaired

    # Быстрая билинейная схема по ортогональному кресту (среднее 4-связных исправных соседей)
    kernel_cross = np.array([[0.0, 1.0, 0.0],
                             [1.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0]], dtype=np.float64)
    valid_mask = (~bad_mask).astype(np.float64)
    weighted_sum = convolve(repaired * valid_mask, kernel_cross, mode='reflect')
    weight_norm = convolve(valid_mask, kernel_cross, mode='reflect')

    # Там, где хотя бы один из 4 прямых соседей исправен
    has_orth = (weight_norm > 0) & bad_mask
    repaired[has_orth] = weighted_sum[has_orth] / weight_norm[has_orth]

    # Если все 4 прямых соседа оказались битыми (редкий кластер), заполняем медианой 5x5
    remaining = bad_mask & (~has_orth)
    if np.any(remaining):
        med5 = median_filter(repaired, size=5, mode='reflect')
        repaired[remaining] = med5[remaining]

    # 2 итерации релаксации Якоби внутри маски для гладкого билинейного сопряжения
    for _ in range(2):
        num = convolve(repaired, kernel_cross / 4.0, mode='reflect')
        repaired[bad_mask] = num[bad_mask]

    return repaired


def inpaint_weighted(image, bad_mask):
    """
    Метод 3: Взвешенная интерполяция (обратно-квадратичные / гауссовы веса 5x5 с маскированием битых соседей
    и робастным подавлением градиентных выбросов).
    """
    repaired = image.copy()
    if not np.any(bad_mask):
        return repaired

    # Ядро 5x5 с весами exp(-r^2 / (2*sigma^2)), где центральный вес = 0
    y, x = np.mgrid[-2:3, -2:3]
    r2 = x ** 2 + y ** 2
    sigma = 1.15
    kernel = np.exp(-r2 / (2.0 * sigma ** 2))
    kernel[2, 2] = 0.0

    valid_mask = (~bad_mask).astype(np.float64)
    # Инициализируем дефекты медианой, чтобы на границах контрастных пятен не было увода
    med_init = median_filter(repaired, size=3, mode='reflect')

    w_sum = convolve(repaired * valid_mask, kernel, mode='reflect')
    w_norm = convolve(valid_mask, kernel, mode='reflect')

    interp_vals = np.where(w_norm > 1e-8, w_sum / np.maximum(w_norm, 1e-8), med_init)
    # Комбинируем взвешенную гауссову интерполяцию с локальной медианой (80% weighted + 20% median)
    # для устойчивости как на спокойной фотосфере, так и на резких границах солнечных пятен/активных областей
    repaired[bad_mask] = 0.80 * interp_vals[bad_mask] + 0.20 * med_init[bad_mask]
    return repaired


def correct_scientific_frame(
    y_frame,
    master_dark,
    g_hat,
    bad_mask_hat,
    method='weighted',
    clip_range=None,
    save_fits_path=None,
    orig_header=None,
    calib_info=None,
    preserve_dtype=np.float32
):
    """
    Полный пайплайн коррекции научного кадра:
    1. Вычитание master_dark и деление на G_hat.
    2. Реконструкция дефектных пикселей выбранным методом ('bilinear', 'median', 'weighted').
    3. Опциональное ограничение динамического диапазона и сохранение в FITS с записью метаданных калибровки.
    """
    s_raw = radiometric_calibrate(y_frame, master_dark, g_hat)

    if method == 'bilinear':
        s_repaired = inpaint_bilinear(s_raw, bad_mask_hat)
    elif method == 'median':
        s_repaired = inpaint_median_neighbors(s_raw, bad_mask_hat)
    elif method == 'weighted':
        s_repaired = inpaint_weighted(s_raw, bad_mask_hat)
    else:
        raise ValueError(f"Unknown inpainting method: {method}")

    if clip_range is not None:
        s_repaired = np.clip(s_repaired, clip_range[0], clip_range[1])

    if save_fits_path:
        hdr = dict(orig_header) if orig_header else {}
        hdr['CAL_DARK'] = True
        hdr['CAL_FLAT'] = True
        hdr['INPAINT'] = method.upper()
        hdr['NBADPIX'] = int(np.sum(bad_mask_hat))
        if calib_info:
            for k, v in calib_info.items():
                hdr[k[:8]] = v
        write_fits(save_fits_path, s_repaired, hdr, preserve_dtype=preserve_dtype)

    return s_repaired, s_raw
