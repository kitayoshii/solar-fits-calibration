"""
detection.py — Модуль этапа 4 (часть 2):
Построение маски дефектных пикселей (bad-pixel mask) и классификация найденных дефектов по типам
исключительно через робастные статистики (медиана, MAD, локальная пространственная медиана соседей).
"""
import numpy as np
from scipy.ndimage import median_filter
from calibration import compute_mad


def detect_bad_pixels(
    g_hat,
    master_dark,
    dark_stack=None,
    flat_stack=None,
    k_gain=6.0,
    k_dark=6.0,
    k_temp=4.5,
    local_win=5
):
    """
    Обнаруживает дефектные пиксели по 4 робастным критериям:
    1. Отклонение чувствительности G_hat от локальной пространственной медианы (> k_gain * MAD_local_G)
       и глобальное падение G_hat (< 0.25) -> выявляет dead и stuck пиксели.
    2. Аномальный темновой сигнал D_hat относительно локальной медианы (> k_dark * MAD_local_D)
       -> выявляет hot и stuck пиксели.
    3. Повышенная временная дисперсия (или робастный разброс по серии dark/flat) относительно медианы разбросов
       (> k_temp * MAD_temp) -> выявляет noisy пиксели.
    4. Нулевая реакция на засветку между dark и flat (|F_med - D_med| ~ 0) -> отделяет stuck от hot/dead.

    Возвращает:
    - bad_mask_hat (bool 2D)
    - defect_types_hat (uint8 2D: 0=good, 1=dead, 2=hot, 3=stuck, 4=noisy)
    - counts_dict (словарь с количеством найденных дефектов каждого типа)
    """
    # 1. Пространственная аномальность G_hat относительно локальной медианы соседей (учитывает виньетирование и low-freq)
    local_med_g = median_filter(g_hat, size=local_win, mode='reflect')
    res_g = g_hat - local_med_g
    mad_res_g = float(compute_mad(res_g))

    # 2. Пространственная аномальность master_dark относительно локальной медианы соседей (учитывает amp glow)
    local_med_d = median_filter(master_dark, size=local_win, mode='reflect')
    res_d = master_dark - local_med_d
    mad_res_d = float(compute_mad(res_d))

    # 3. Временная дисперсия / СКО по серии кадров (если N >= 4)
    has_temporal = (dark_stack is not None and dark_stack.shape[0] >= 4)
    if has_temporal:
        temp_std_d = np.std(dark_stack, axis=0)
        med_std_d = float(np.median(temp_std_d))
        mad_std_d = float(compute_mad(temp_std_d))
        is_noisy = temp_std_d > (med_std_d + k_temp * max(mad_std_d, 1e-4))
    else:
        # При малом числе кадров оцениваем noisy по высокочастотным выбросам в dark + flat
        is_noisy = np.abs(res_d) > (k_dark * 1.15 * max(mad_res_d, 1e-4))

    # Проверка реакции на свет для разделения stuck и dead/hot
    if flat_stack is not None:
        flat_med = np.median(flat_stack, axis=0)
        light_response = np.abs(flat_med - master_dark)
        med_resp = float(np.median(light_response))
        no_light_response = light_response < (0.015 * med_resp)
    else:
        no_light_response = g_hat < 0.01

    # Классификация:
    # Stuck (3): высокий темновой отклик D_hat И одновременно околонулевой отклик на свет G_hat ~ 0
    is_high_d = res_d > (k_dark * max(mad_res_d, 1e-4))
    is_low_g = (res_g < -(k_gain * max(mad_res_g, 1e-4))) | (g_hat < 0.25)
    is_high_g = res_g > (k_gain * max(mad_res_g, 1e-4))

    is_stuck = is_high_d & (no_light_response | (g_hat < 0.05))
    is_dead = is_low_g & (~is_stuck)
    is_hot = is_high_d & (~is_stuck)
    is_noisy_clean = (is_noisy | is_high_g) & (~is_stuck) & (~is_dead) & (~is_hot)

    defect_types_hat = np.zeros(g_hat.shape, dtype=np.uint8)
    defect_types_hat[is_noisy_clean] = 4
    defect_types_hat[is_hot] = 2
    defect_types_hat[is_dead] = 1
    defect_types_hat[is_stuck] = 3

    bad_mask_hat = defect_types_hat > 0
    counts = {
        'dead': int(np.sum(defect_types_hat == 1)),
        'hot': int(np.sum(defect_types_hat == 2)),
        'stuck': int(np.sum(defect_types_hat == 3)),
        'noisy': int(np.sum(defect_types_hat == 4)),
        'total_bad': int(np.sum(bad_mask_hat))
    }
    return bad_mask_hat, defect_types_hat, counts
