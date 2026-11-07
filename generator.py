"""
generator.py — Модуль этапа 1, 2 и 3:
- Загрузка и подготовка реальных солнечных FITS-данных (Этап 1)
- Генератор истинных карт фотоприемника G(x,y), D(x,y) и маски дефектов (Этап 2)
- Генерация синтетических серий flat-field и dark-frame, а также искаженных научных кадров (Этап 3 и 6)
"""
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from fits_utils import read_fits, write_fits


# ==========================================
# ЭТАП 1: Загрузка и подготовка реальных данных
# ==========================================
def inspect_and_prepare_solar_fits(
    input_dir='./data/solar',
    norm_dir='./data/normalized',
    preview_dir='./results/previews',
    summary_csv='./results/tables/stage1_fits_summary.csv'
):
    """
    Читает все FITS из input_dir, выводит статистику и ключевые поля header,
    приводит к float64, сохраняет нормализованные копии и строит PNG-превью.
    """
    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    fits_paths = sorted(glob.glob(os.path.join(input_dir, '*.fits')))
    records = []
    solar_frames = []

    for fp in fits_paths:
        fname = os.path.basename(fp)
        raw_data, hdr = read_fits(fp)
        orig_dtype = str(raw_data.dtype)
        shape = raw_data.shape
        nan_count = int(np.isnan(raw_data).sum())

        # Приведение к float64 и замена NaN (если есть) на медиану
        data_f64 = raw_data.astype(np.float64)
        if nan_count > 0:
            med_val = float(np.nanmedian(data_f64))
            data_f64 = np.nan_to_num(data_f64, nan=med_val)

        d_min = float(np.min(data_f64))
        d_max = float(np.max(data_f64))
        d_med = float(np.median(data_f64))
        d_mean = float(np.mean(data_f64))
        d_std = float(np.std(data_f64))

        # Нормализованная копия S(x,y) в диапазон [0, 1] и рабочий диапазон АЦП [0, 1000] DN
        # В AIA 304A присутствуют небольшие отрицательные значения из-за шума чтения в космосе (-4..-1 DN),
        # сдвигаем пьедестал так, чтобы фон неба был >= 0.
        s_shifted = np.clip(data_f64 - d_min, 0.0, None)
        s_norm_01 = s_shifted / (np.max(s_shifted) + 1e-12)

        # Сохраняем нормализованную копию
        np.save(os.path.join(norm_dir, fname.replace('.fits', '_norm.npy')), s_norm_01)

        # Построение PNG-превью с указанием min/max и перцентилей для наглядности солнечной короны
        vmin_p1, vmax_p99 = np.percentile(data_f64, [1.0, 99.5])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=120)
        im0 = axes[0].imshow(data_f64, cmap='inferno', origin='lower', vmin=d_min, vmax=d_max)
        axes[0].set_title(f"{fname}\nЛинейная шкала [min={d_min:.1f}, max={d_max:.1f}]", fontsize=10)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(np.arcsinh(np.clip(data_f64, 0, None)), cmap='magma', origin='lower')
        axes[1].set_title(f"Шкала arcsinh(S) (контраст активных областей)\nmedian={d_med:.2f}, std={d_std:.2f}", fontsize=10)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        preview_path = os.path.join(preview_dir, fname.replace('.fits', '_preview.png'))
        plt.savefig(preview_path)
        plt.close(fig)

        rec = {
            'filename': fname,
            'shape': f"{shape[0]}x{shape[1]}",
            'dtype': orig_dtype,
            'BITPIX': hdr.get('BITPIX', None),
            'EXPTIME': round(float(hdr.get('EXPTIME', 0.0)), 6),
            'WAVELNTH': hdr.get('WAVELNTH', None),
            'BSCALE': hdr.get('BSCALE', 1.0),
            'BZERO': hdr.get('BZERO', 0.0),
            'min': round(d_min, 3),
            'max': round(d_max, 3),
            'median': round(d_med, 3),
            'mean': round(d_mean, 3),
            'std': round(d_std, 3),
            'nan_count': nan_count,
            'T_OBS': hdr.get('T_OBS', '')
        }
        records.append(rec)
        solar_frames.append((fname, data_f64, s_norm_01, hdr))

    df = pd.DataFrame(records)
    df.to_csv(summary_csv, index=False)
    return df, solar_frames


# ==========================================
# ЭТАП 2: Генератор истинных карт фотоприемника
# ==========================================
def make_prnu(size, sigma_g=0.02, rng=None):
    """
    Пиксельная PRNU (Photo Response Non-Uniformity):
    гауссово отклонение со средним 1 и СКО sigma_g, нормировка среднего к 1.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    prnu = rng.normal(loc=1.0, scale=sigma_g, size=size)
    prnu = np.clip(prnu, 0.2, 2.5)
    prnu /= np.mean(prnu)
    return prnu.astype(np.float64)


def make_lowfreq(size, amplitude=0.05, scale=8, rng=None):
    """
    Низкочастотное поле чувствительности:
    случайная матрица малого размера (scale x scale) -> масштабирование -> Gaussian-сглаживание -> нормировка к среднему 1.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    h, w = size
    small_grid = rng.normal(loc=0.0, scale=1.0, size=(scale, scale))
    zoom_factors = (h / scale, w / scale)
    up_field = zoom(small_grid, zoom_factors, order=3)
    smooth_field = gaussian_filter(up_field, sigma=max(h, w) / (2.0 * scale))
    # Нормируем размах к заданной амплитуде (STD = amplitude)
    if np.std(smooth_field) > 1e-12:
        smooth_field = smooth_field / np.std(smooth_field) * amplitude
    lowfreq = 1.0 + smooth_field
    lowfreq = np.clip(lowfreq, 0.3, 2.0)
    lowfreq /= np.mean(lowfreq)
    return lowfreq.astype(np.float64)


def make_radial(size, amplitude=0.12):
    """
    Радиальная составляющая (виньетирование / градиент к краю):
    плавный спад к углам матрицы по закону 1 - amplitude * (r / r_max)^2, нормировка к среднему 1.
    """
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    r2_max = cy ** 2 + cx ** 2
    radial = 1.0 - amplitude * (r2 / r2_max)
    radial = np.clip(radial, 0.2, 2.0)
    radial /= np.mean(radial)
    return radial.astype(np.float64)


def make_dark_offset(size, level=15.0, sigma=2.5, amp_glow=4.0, rng=None):
    """
    Пространственно неоднородный темновой сигнал D(x,y):
    базовый пьедестал `level`, пиксельная неоднородность `sigma` и слабый градиент (amp glow) к углу чтения.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    h, w = size
    y, x = np.ogrid[:h, :w]
    # Слабый экспоненциальный разогрев усилителя в правом верхнем углу + плавный наклон
    glow = amp_glow * np.exp(-((h - 1 - y) ** 2 + (w - 1 - x) ** 2) / (2.0 * (0.25 * h) ** 2))
    pixel_dark = rng.normal(loc=level, scale=sigma, size=size)
    dark = np.clip(pixel_dark + glow, 0.0, None)
    return dark.astype(np.float64)


def make_bad_mask(size, fractions=None, rng=None):
    """
    Генерация маски дефектных пикселей и карты типов дефектов:
    0 — нормальный пиксель
    1 — dead (G ≈ 0, например 0.02)
    2 — hot (аномально высокий темновой ток D)
    3 — stuck (фиксированный отклик, не зависит от L)
    4 — noisy (повышенная временная дисперсия шума, RTS/телеграфный шум)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if fractions is None:
        fractions = {'dead': 0.0005, 'hot': 0.0005, 'stuck': 0.0002, 'noisy': 0.0003}

    h, w = size
    n_total = h * w
    defect_type_map = np.zeros(n_total, dtype=np.uint8)

    # Исключаем самую границу (2 пикселя от края) для чистоты оценки интерполяции, либо распределяем по всей площади
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)

    type_codes = [('dead', 1), ('hot', 2), ('stuck', 3), ('noisy', 4)]
    cursor = 0
    for name, code in type_codes:
        frac = float(fractions.get(name, 0.0))
        count = int(round(frac * n_total))
        if count > 0 and cursor + count <= n_total:
            selected = all_indices[cursor:cursor + count]
            defect_type_map[selected] = code
            cursor += count

    defect_type_map = defect_type_map.reshape(size)
    bad_mask = (defect_type_map > 0)
    return bad_mask, defect_type_map


def build_ground_truth_detector(
    size=(1024, 1024),
    sigma_g=0.02,
    lowfreq_amp=0.04,
    lowfreq_scale=8,
    radial_amp=0.10,
    dark_level=15.0,
    dark_sigma=2.0,
    fractions=None,
    dynamic_range=1000.0,
    seed=42,
    save_npz_path=None,
    save_json_path=None
):
    """
    Собирает полную истинную модель фотоприемника:
    G_true(x,y), D_true(x,y), bad_mask, defect_types, stuck_values.
    Сохраняет истинные карты в .npz отдельно от входных данных алгоритма.
    """
    rng = np.random.default_rng(seed)

    g_prnu = make_prnu(size, sigma_g=sigma_g, rng=rng)
    g_lowfreq = make_lowfreq(size, amplitude=lowfreq_amp, scale=lowfreq_scale, rng=rng)
    g_radial = make_radial(size, amplitude=radial_amp)

    g_true = g_prnu * g_lowfreq * g_radial
    g_true /= np.mean(g_true)

    d_true = make_dark_offset(size, level=dark_level, sigma=dark_sigma, rng=rng)
    bad_mask, defect_types = make_bad_mask(size, fractions=fractions, rng=rng)

    # Применяем физические свойства дефектов к G_true и D_true:
    # 1: dead -> чувствительность почти нулевая (0.01..0.05)
    dead_mask = (defect_types == 1)
    g_true[dead_mask] = rng.uniform(0.01, 0.05, size=int(np.sum(dead_mask)))

    # 2: hot -> высокий темновой сигнал (от 25% до 65% динамического диапазона)
    hot_mask = (defect_types == 2)
    d_true[hot_mask] = rng.uniform(0.25 * dynamic_range, 0.65 * dynamic_range, size=int(np.sum(hot_mask)))

    # 3: stuck -> пиксель залип на фиксированном уровне (не зависит от освещения -> G=0, D=const)
    stuck_mask = (defect_types == 3)
    stuck_vals = rng.uniform(0.35 * dynamic_range, 0.80 * dynamic_range, size=int(np.sum(stuck_mask)))
    g_true[stuck_mask] = 0.0
    d_true[stuck_mask] = stuck_vals

    # 4: noisy -> повышенная дисперсия чтения (в 12 раз выше обычного read_noise)
    # Сами G и D в noisy-пикселях имеют умеренный сдвиг, чтобы они проявлялись именно через временную дисперсию
    noisy_mask = (defect_types == 4)

    # Перенормируем G_true на исправных пикселях так, чтобы среднее по рабочим пикселям было 1.0
    good_mask = ~bad_mask
    g_true /= np.mean(g_true[good_mask])

    gt_dict = {
        'G_true': g_true,
        'G_prnu': g_prnu,
        'G_lowfreq': g_lowfreq,
        'G_radial': g_radial,
        'D_true': d_true,
        'bad_mask': bad_mask,
        'defect_types': defect_types,
        'dynamic_range': np.array(dynamic_range, dtype=np.float64),
        'seed': np.array(seed, dtype=np.int64)
    }

    if save_npz_path:
        os.makedirs(os.path.dirname(save_npz_path), exist_ok=True)
        np.savez_compressed(save_npz_path, **gt_dict)

    if save_json_path:
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        meta = {
            'size': list(size),
            'sigma_g': sigma_g,
            'lowfreq_amp': lowfreq_amp,
            'lowfreq_scale': lowfreq_scale,
            'radial_amp': radial_amp,
            'dark_level': dark_level,
            'dark_sigma': dark_sigma,
            'fractions': fractions if fractions else {'dead': 0.0005, 'hot': 0.0005, 'stuck': 0.0002, 'noisy': 0.0003},
            'dynamic_range': dynamic_range,
            'seed': seed,
            'defect_counts': {
                'dead': int(np.sum(dead_mask)),
                'hot': int(np.sum(hot_mask)),
                'stuck': int(np.sum(stuck_mask)),
                'noisy': int(np.sum(noisy_mask)),
                'total_bad': int(np.sum(bad_mask))
            }
        }
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return gt_dict


# ==========================================
# ЭТАП 3: Физическая модель регистрации кадра и генерация серий
# ==========================================
def simulate_frame(
    signal,
    gt_dict,
    read_noise=2.0,
    gain_e_per_dn=2.0,
    noisy_multiplier=12.0,
    rng=None
):
    """
    Физическая модель формирования кадра:
    Y(x,y) = clip{ G(x,y)*S(x,y) + D(x,y) + N(x,y) }
    где N включает пуассоновский фотонный шум (зависящий от уровня сигнала G*S)
    и аддитивный гауссов шум чтения (повышенный в noisy-пикселях).
    """
    if rng is None:
        rng = np.random.default_rng()

    g_true = gt_dict['G_true']
    d_true = gt_dict['D_true']
    defect_types = gt_dict['defect_types']
    dyn_range = float(gt_dict['dynamic_range'])

    expected_light_dn = np.clip(g_true * signal, 0.0, None)

    # Пуассоновский фотонный шум (в фотоэлектронах с пересчетом обратно в DN)
    if gain_e_per_dn is not None and gain_e_per_dn > 0:
        expected_e = np.clip(expected_light_dn * gain_e_per_dn, 0.0, 1e8)
        noisy_light_dn = rng.poisson(expected_e).astype(np.float64) / gain_e_per_dn
    else:
        noisy_light_dn = expected_light_dn.copy()

    # Гауссов шум чтения (в noisy-пикселях СКО увеличено в noisy_multiplier раз)
    sigma_map = np.full(g_true.shape, read_noise, dtype=np.float64)
    sigma_map[defect_types == 4] = read_noise * noisy_multiplier
    read_noise_realization = rng.normal(loc=0.0, scale=sigma_map)

    # Stuck пиксели возвращают строго фиксированный уровень D_true без шума или с минимальным квантованием
    raw_frame = noisy_light_dn + d_true + read_noise_realization
    stuck_mask = (defect_types == 3)
    if np.any(stuck_mask):
        raw_frame[stuck_mask] = d_true[stuck_mask]

    clipped_frame = np.clip(raw_frame, 0.0, dyn_range)
    return clipped_frame


def generate_dark_series(
    n_frames,
    gt_dict,
    read_noise=2.0,
    seed=100,
    save_dir=None,
    exptime=2.90083
):
    """
    Генерирует серию dark-frame: B_k(x,y) = D(x,y) + N_k(x,y).
    При указании save_dir сохраняет кадры в FITS.
    """
    rng = np.random.default_rng(seed)
    size = gt_dict['G_true'].shape
    zero_signal = np.zeros(size, dtype=np.float64)
    dark_stack = np.empty((n_frames, size[0], size[1]), dtype=np.float64)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for k in range(n_frames):
        frame = simulate_frame(
            zero_signal,
            gt_dict,
            read_noise=read_noise,
            gain_e_per_dn=0.0,  # Нет светового потока
            rng=rng
        )
        dark_stack[k] = frame
        if save_dir:
            hdr = {
                'IMG_TYPE': 'DARK',
                'EXPTIME': float(exptime),
                'FRAMENUM': k + 1,
                'NFRAMES': n_frames,
                'SEED': int(seed + k),
                'RDNOISE': float(read_noise)
            }
            write_fits(os.path.join(save_dir, f"dark_{k+1:02d}.fits"), frame, hdr, preserve_dtype=np.float32)

    return dark_stack


def generate_flat_series(
    n_frames,
    level_fraction,
    gt_dict,
    read_noise=2.0,
    gain_e_per_dn=2.0,
    drift_sigma=0.0,
    illumination_grad=0.0,
    seed=200,
    save_dir=None,
    exptime=2.90083
):
    """
    Генерирует серию flat-field кадров:
    F_k(x,y) = clip{ G(x,y)*L_k(x,y) + D(x,y) + N_k(x,y) }
    Поддерживает:
    - уровень L_k в долях динамического диапазона (например 0.2, 0.4, 0.6, 0.8),
    - опциональный межкадровый дрейф источника drift_sigma,
    - опциональную неоднородность самого источника плоского поля illumination_grad.
    """
    rng = np.random.default_rng(seed)
    size = gt_dict['G_true'].shape
    dyn_range = float(gt_dict['dynamic_range'])
    base_level = level_fraction * dyn_range

    # Если задана внешняя неравномерность осветителя (для эксперимента №7)
    if illumination_grad > 0.0:
        h, w = size
        y, x = np.mgrid[:h, :w]
        illum_profile = 1.0 + illumination_grad * ((x - w / 2.0) / (w / 2.0) * 0.7 + (y - h / 2.0) / (h / 2.0) * 0.7)
        illum_profile /= np.mean(illum_profile)
    else:
        illum_profile = np.ones(size, dtype=np.float64)

    flat_stack = np.empty((n_frames, size[0], size[1]), dtype=np.float64)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for k in range(n_frames):
        drift_factor = 1.0 + (rng.normal(0.0, drift_sigma) if drift_sigma > 0 else 0.0)
        l_k_val = base_level * drift_factor
        l_k_field = l_k_val * illum_profile

        frame = simulate_frame(
            l_k_field,
            gt_dict,
            read_noise=read_noise,
            gain_e_per_dn=gain_e_per_dn,
            rng=rng
        )
        flat_stack[k] = frame
        if save_dir:
            hdr = {
                'IMG_TYPE': 'FLAT',
                'EXPTIME': float(exptime),
                'FLAT_LVL': float(level_fraction),
                'LK_VAL': float(l_k_val),
                'FRAMENUM': k + 1,
                'NFRAMES': n_frames,
                'SEED': int(seed + k),
                'DRIFT': float(drift_sigma)
            }
            write_fits(os.path.join(save_dir, f"flat_L{int(level_fraction*100):02d}_{k+1:02d}.fits"), frame, hdr, preserve_dtype=np.float32)

    return flat_stack
