"""
metrics.py — Модуль этапа 6:
Вычисление полного набора метрик качества калибровки и реконструкции научных солнечных кадров:
- RMSE и MAE между истинной G и оценённой G_hat (по рабочим пикселям и по всему полю);
- Максимальная и 95-процентильная ошибка оценки чувствительности;
- Остаточная пространственная неравномерность на тестовом равномерном поле;
- RMSE и MAE между исходным S и восстановленным S_hat;
- PSNR и SSIM;
- Ошибка восстановления локального контраста и относительных яркостей выбранных областей;
- Precision, Recall, F1-score для bad-pixel mask (в целом и по типам дефектов);
- Доля дефектов, успешно восстановленных без заметного локального артефакта;
- Зональная оценка: (1) спокойная фотосфера/хромосфера, (2) границы пятен/корональных дыр (высокий градиент),
  (3) яркие активные области, (4) участки с малым локальным контрастом.
"""
import numpy as np
from scipy.ndimage import uniform_filter, sobel, median_filter
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from calibration import compute_mad


def evaluate_gain_metrics(g_true, g_hat, bad_mask_true=None):
    """
    Оценивает точность восстановления карты относительной чувствительности G_hat(x,y)
    на исправных пикселях (так как битые пиксели заменяются интерполяцией по маске).
    Также оценивает остаточную пространственную неравномерность на тестовом равномерном поле.
    """
    if bad_mask_true is not None:
        valid = ~bad_mask_true
    else:
        valid = np.ones_like(g_true, dtype=bool)

    # Приводим к единой нормировке по медиане исправных пикселей
    gt_norm = g_true / np.median(g_true[valid])
    gh_norm = g_hat / np.median(g_hat[valid])

    abs_err = np.abs(gt_norm[valid] - gh_norm[valid])
    rmse_g = float(np.sqrt(np.mean(abs_err ** 2)))
    mae_g = float(np.mean(abs_err))
    max_err_g = float(np.max(abs_err))
    p95_err_g = float(np.percentile(abs_err, 95.0))

    # Остаточная пространственная неравномерность на тестовом равномерном поле U_test = G_true * L_0 / G_hat
    # В идеале отношение G_true / G_hat на исправных пикселях должно иметь минимальное СКО (относительно 1.0)
    uniform_corrected = gt_norm[valid] / np.maximum(gh_norm[valid], 0.05)
    residual_nonuniformity_std = float(np.std(uniform_corrected))
    residual_nonuniformity_mad = float(compute_mad(uniform_corrected))

    return {
        'G_RMSE': rmse_g,
        'G_MAE': mae_g,
        'G_MaxErr': max_err_g,
        'G_P95Err': p95_err_g,
        'Residual_NonUniformity_STD': residual_nonuniformity_std,
        'Residual_NonUniformity_MAD': residual_nonuniformity_mad
    }


def evaluate_mask_metrics(bad_mask_true, bad_mask_hat, defect_types_true=None):
    """
    Вычисляет Precision, Recall и F1 для маски дефектных пикселей, а также полноту обнаружения (Recall)
    отдельно по каждому физическому типу дефекта (dead, hot, stuck, noisy).
    """
    y_true = bad_mask_true.astype(bool)
    y_pred = bad_mask_hat.astype(bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    res = {
        'Mask_TP': tp,
        'Mask_FP': fp,
        'Mask_FN': fn,
        'Mask_Precision': precision,
        'Mask_Recall': recall,
        'Mask_F1': f1
    }

    if defect_types_true is not None:
        for name, code in [('dead', 1), ('hot', 2), ('stuck', 3), ('noisy', 4)]:
            t_mask = (defect_types_true == code)
            n_t = int(np.sum(t_mask))
            if n_t > 0:
                res[f'Recall_{name}'] = float(np.sum(t_mask & y_pred) / n_t)
            else:
                res[f'Recall_{name}'] = 1.0

    return res


def evaluate_image_metrics(s_true, s_hat, bad_mask_true=None, dynamic_range=None):
    """
    Оценивает качество восстановленного научного изображения S_hat(x,y) относительно эталона S(x,y):
    - RMSE, MAE (по всему кадру и в точках дефектов)
    - PSNR, SSIM
    - Ошибка восстановления локального контраста и относительной яркости областей
    - Доля дефектов, восстановленных без заметного локального артефакта
    """
    if dynamic_range is None:
        dynamic_range = float(np.max(s_true) - np.min(s_true))
        if dynamic_range < 1e-6:
            dynamic_range = 1.0

    diff = s_hat - s_true
    rmse_s = float(np.sqrt(np.mean(diff ** 2)))
    mae_s = float(np.mean(np.abs(diff)))

    s_true_clipped = np.clip(s_true, 0.0, dynamic_range)
    s_hat_clipped = np.clip(s_hat, 0.0, dynamic_range)

    psnr_val = float(skimage_psnr(s_true_clipped, s_hat_clipped, data_range=dynamic_range))
    ssim_val = float(skimage_ssim(s_true_clipped, s_hat_clipped, data_range=dynamic_range))

    # Ошибка восстановления локального контраста:
    # Локальный контраст C(x,y) = std_local(7x7) / (mean_local(7x7) + eps)
    eps = 0.01 * dynamic_range
    mean_t = uniform_filter(s_true, size=7)
    sq_mean_t = uniform_filter(s_true ** 2, size=7)
    std_t = np.sqrt(np.maximum(sq_mean_t - mean_t ** 2, 0.0))
    contrast_t = std_t / ( np.abs(mean_t) + eps )

    mean_h = uniform_filter(s_hat, size=7)
    sq_mean_h = uniform_filter(s_hat ** 2, size=7)
    std_h = np.sqrt(np.maximum(sq_mean_h - mean_h ** 2, 0.0))
    contrast_h = std_h / ( np.abs(mean_h) + eps )

    local_contrast_mae = float(np.mean(np.abs(contrast_h - contrast_t)))

    # Доля дефектов, успешно восстановленных без заметного локального артефакта:
    # Артефакт отсутствует, если остаточная ошибка в точке дефекта не превышает 3 * локальный MAD окрестности (или 2% шкалы)
    if bad_mask_true is not None and np.any(bad_mask_true):
        local_med = median_filter(s_true, size=5)
        local_mad_map = median_filter(np.abs(s_true - local_med), size=7) * 1.4826
        tol_map = np.maximum(3.0 * local_mad_map, 0.015 * dynamic_range)
        defect_errors = np.abs(s_hat[bad_mask_true] - s_true[bad_mask_true])
        artifact_free_ratio = float(np.mean(defect_errors <= tol_map[bad_mask_true]))
        defect_rmse = float(np.sqrt(np.mean(defect_errors ** 2)))
    else:
        artifact_free_ratio = 1.0
        defect_rmse = 0.0

    return {
        'S_RMSE': rmse_s,
        'S_MAE': mae_s,
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'Local_Contrast_MAE': local_contrast_mae,
        'Defect_Only_RMSE': defect_rmse,
        'Artifact_Free_Defects_Ratio': artifact_free_ratio
    }


def segment_solar_regions(s_true):
    """
    Автоматически сегментирует солнечный кадр на 4 характерные физические зоны:
    1. 'quiet_photosphere' — спокойная фотосфера/хромосфера на солнечном диске (средняя яркость, умеренный градиент).
    2. 'spot_boundaries' — границы пятен / корональных петель и лимба (высокий пространственный градиент Sobel > P85).
    3. 'bright_active_regions' — яркие активные области и вспышечные узлы (интенсивность > P92).
    4. 'low_local_contrast' — участки со сверхмалым локальным контрастом (спокойный фон и однородные области короны, локальное СКО < P30).
    """
    gx = sobel(s_true, axis=1)
    gy = sobel(s_true, axis=0)
    grad_mag = np.hypot(gx, gy)

    mean_loc = uniform_filter(s_true, size=7)
    sq_loc = uniform_filter(s_true ** 2, size=7)
    std_loc = np.sqrt(np.maximum(sq_loc - mean_loc ** 2, 0.0))

    p30_int, p85_int, p92_int = np.percentile(s_true, [30.0, 85.0, 92.0])
    p85_grad = np.percentile(grad_mag, 85.0)
    p30_std = np.percentile(std_loc, 30.0)

    bright_active = s_true >= p92_int
    spot_boundaries = (grad_mag >= p85_grad) & (~bright_active)
    low_contrast = (std_loc <= p30_std) & (~bright_active) & (~spot_boundaries)
    quiet_photo = (s_true > p30_int) & (s_true < p85_int) & (~spot_boundaries) & (~low_contrast)

    return {
        'quiet_photosphere': quiet_photo,
        'spot_boundaries': spot_boundaries,
        'bright_active_regions': bright_active,
        'low_local_contrast': low_contrast
    }


def evaluate_regional_metrics(s_true, s_hat):
    """
    Вычисляет RMSE, MAE и ошибку относительной яркости по 4 характерным солнечным зонам.
    """
    regions = segment_solar_regions(s_true)
    reg_results = {}
    global_mean = float(np.mean(s_true)) + 1e-8

    for rname, rmask in regions.items():
        if np.any(rmask):
            diff = s_hat[rmask] - s_true[rmask]
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            # Относительная яркость области (отношение средней яркости зоны к средней яркости кадра)
            rel_b_true = float(np.mean(s_true[rmask]) / global_mean)
            rel_b_hat = float(np.mean(s_hat[rmask]) / (float(np.mean(s_hat)) + 1e-8))
            rel_b_err = abs(rel_b_hat - rel_b_true) / max(rel_b_true, 1e-6)
        else:
            rmse, mae, rel_b_err = 0.0, 0.0, 0.0

        reg_results[rname] = {
            'RMSE': rmse,
            'MAE': mae,
            'RelBrightness_RelErr': float(rel_b_err),
            'Pixel_Count': int(np.sum(rmask))
        }
    return reg_results, regions
