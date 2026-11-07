"""
experiments.py — Модуль этапа 7:
Автоматизированный прогон полного набора минимальных экспериментов (№1–7 и №9 — комплексная модель)
по исследованию устойчивости калибровки фотоприемника и восстановления солнечных FITS-кадров.
Сохраняет все таблицы результатов в CSV и возвращает сводный DataFrame.
"""
import os
import numpy as np
import pandas as pd
from generator import (
    build_ground_truth_detector,
    generate_dark_series,
    generate_flat_series,
    simulate_frame
)
from calibration import build_master_dark, build_master_flat
from detection import detect_bad_pixels
from correction import correct_scientific_frame
from metrics import evaluate_gain_metrics, evaluate_mask_metrics, evaluate_image_metrics


def run_single_configuration(
    s_ref,
    size=(1024, 1024),
    sigma_g=0.02,
    lowfreq_amp=0.04,
    radial_amp=0.10,
    dark_level=15.0,
    dark_sigma=2.0,
    fractions=None,
    n_dark=16,
    n_flat=16,
    flat_level=0.60,
    read_noise=2.0,
    gain_e_per_dn=2.0,
    drift_sigma=0.0,
    illumination_grad=0.0,
    inpaint_method='weighted',
    dynamic_range=1000.0,
    seed=42
):
    """
    Выполняет один полный цикл:
    Генерация GT -> Генерация Dark/Flat -> Оценка G_hat, D_hat, bad_mask_hat ->
    Искажение научного кадра Y -> Восстановление S_hat -> Расчет всех метрик.
    """
    gt = build_ground_truth_detector(
        size=size,
        sigma_g=sigma_g,
        lowfreq_amp=lowfreq_amp,
        radial_amp=radial_amp,
        dark_level=dark_level,
        dark_sigma=dark_sigma,
        fractions=fractions,
        dynamic_range=dynamic_range,
        seed=seed
    )

    dark_stack = generate_dark_series(
        n_frames=n_dark,
        gt_dict=gt,
        read_noise=read_noise,
        seed=seed + 100
    )

    flat_stack = generate_flat_series(
        n_frames=n_flat,
        level_fraction=flat_level,
        gt_dict=gt,
        read_noise=read_noise,
        gain_e_per_dn=gain_e_per_dn,
        drift_sigma=drift_sigma,
        illumination_grad=illumination_grad,
        seed=seed + 200
    )

    master_dark, _ = build_master_dark(dark_stack)
    g_hat, _, _ = build_master_flat(flat_stack, master_dark)
    bad_mask_hat, defect_types_hat, counts_hat = detect_bad_pixels(
        g_hat, master_dark, dark_stack=dark_stack, flat_stack=flat_stack
    )

    rng_sci = np.random.default_rng(seed + 500)
    y_sci = simulate_frame(
        s_ref,
        gt,
        read_noise=read_noise,
        gain_e_per_dn=gain_e_per_dn,
        rng=rng_sci
    )

    s_hat, _ = correct_scientific_frame(
        y_sci,
        master_dark,
        g_hat,
        bad_mask_hat,
        method=inpaint_method,
        clip_range=(0.0, dynamic_range)
    )

    m_gain = evaluate_gain_metrics(gt['G_true'], g_hat, gt['bad_mask'])
    m_mask = evaluate_mask_metrics(gt['bad_mask'], bad_mask_hat, gt['defect_types'])
    m_img = evaluate_image_metrics(s_ref, s_hat, gt['bad_mask'], dynamic_range=dynamic_range)

    res = {}
    res.update(m_gain)
    res.update(m_mask)
    res.update(m_img)
    res['Detected_Bad_Total'] = counts_hat['total_bad']
    res['True_Bad_Total'] = int(np.sum(gt['bad_mask']))
    return res, gt, master_dark, g_hat, bad_mask_hat, y_sci, s_hat


def run_all_minimum_experiments(s_ref_full, out_dir='./results/tables', work_size=(512, 512)):
    """
    Запускает эксперименты №1–7 и №9 (комплексная модель).
    Для скорости серийных прогонов (более 25 конфигураций) использует центральный кроп work_size=(512, 512)
    с солнечным диском и активными областями, а базовый эталонный эксперимент №9 также проверяется на полном 1024x1024.
    """
    os.makedirs(out_dir, exist_ok=True)
    h_full, w_full = s_ref_full.shape
    ch, cw = work_size
    y0, x0 = (h_full - ch) // 2, (w_full - cw) // 2
    s_crop = s_ref_full[y0:y0 + ch, x0:x0 + cw].copy()
    dyn_range = float(np.max(s_ref_full))

    all_rows = []

    def log_row(exp_id, exp_name, param_name, param_val, metrics_dict):
        row = {
            'Exp_ID': exp_id,
            'Experiment': exp_name,
            'Varying_Factor': param_name,
            'Factor_Value': str(param_val),
        }
        row.update(metrics_dict)
        all_rows.append(row)
        return row

    # Эксперимент №1: Число кадров плоского поля N_flat (1, 4, 8, 16, 32)
    exp1_rows = []
    for n_f in [1, 4, 8, 16, 32]:
        m, *_ = run_single_configuration(
            s_crop, size=work_size, n_flat=n_f, n_dark=16, flat_level=0.60, dynamic_range=dyn_range, seed=100 + n_f
        )
        exp1_rows.append(log_row('Exp_1', 'Число кадров Flat (N_flat)', 'N_flat', n_f, m))
    pd.DataFrame(exp1_rows).to_csv(os.path.join(out_dir, 'exp1_n_flat.csv'), index=False)

    # Эксперимент №2: Уровень освещённости Flat-кадров L_k (20%, 40%, 60%, 80%, 95% - с частичным насыщением)
    exp2_rows = []
    for lvl in [0.20, 0.40, 0.60, 0.80, 0.95]:
        m, *_ = run_single_configuration(
            s_crop, size=work_size, n_flat=16, n_dark=16, flat_level=lvl, dynamic_range=dyn_range, seed=200 + int(lvl*100)
        )
        exp2_rows.append(log_row('Exp_2', 'Уровень яркости Flat (L_k)', 'Flat_Level_%', int(lvl * 100), m))
    pd.DataFrame(exp2_rows).to_csv(os.path.join(out_dir, 'exp2_flat_level.csv'), index=False)

    # Эксперимент №3: Амплитуда мелкомасштабной PRNU sigma_G (0.5%, 1%, 2%, 5%, 10%)
    exp3_rows = []
    for sg in [0.005, 0.01, 0.02, 0.05, 0.10]:
        m, *_ = run_single_configuration(
            s_crop, size=work_size, sigma_g=sg, n_flat=16, n_dark=16, flat_level=0.60, dynamic_range=dyn_range, seed=300 + int(sg*1000)
        )
        exp3_rows.append(log_row('Exp_3', 'Амплитуда PRNU (sigma_G)', 'sigma_G', sg, m))
    pd.DataFrame(exp3_rows).to_csv(os.path.join(out_dir, 'exp3_sigma_g.csv'), index=False)

    # Эксперимент №4: Крупномасштабная и радиальная неравномерность (виньетирование 0%, 5%, 10%, 20%, 30%)
    exp4_rows = []
    for rad in [0.00, 0.05, 0.10, 0.20, 0.30]:
        m, *_ = run_single_configuration(
            s_crop, size=work_size, radial_amp=rad, lowfreq_amp=rad*0.4, n_flat=16, n_dark=16, dynamic_range=dyn_range, seed=400 + int(rad*100)
        )
        exp4_rows.append(log_row('Exp_4', 'Виньетирование и Low-Freq', 'Radial_Amp', rad, m))
    pd.DataFrame(exp4_rows).to_csv(os.path.join(out_dir, 'exp4_vignetting.csv'), index=False)

    # Эксперимент №5: Темновой ток и шум чтения (Read Noise = 0.5, 2.0, 5.0, 10.0 DN при N_dark=16 и без вычитания Dark)
    exp5_rows = []
    for rn in [0.5, 2.0, 5.0, 10.0]:
        m, *_ = run_single_configuration(
            s_crop, size=work_size, read_noise=rn, dark_level=15.0 + rn, n_flat=16, n_dark=16, dynamic_range=dyn_range, seed=500 + int(rn*10)
        )
        exp5_rows.append(log_row('Exp_5', 'Уровень шума чтения и Dark', 'Read_Noise_DN', rn, m))
    pd.DataFrame(exp5_rows).to_csv(os.path.join(out_dir, 'exp5_dark_readnoise.csv'), index=False)

    # Эксперимент №6: Доля дефектных пикселей (0.05%, 0.15%, 0.5%, 1.0%, 2.0%)
    exp6_rows = []
    for total_frac in [0.0005, 0.0015, 0.0050, 0.0100, 0.0200]:
        fracs = {
            'dead': total_frac * 0.35,
            'hot': total_frac * 0.35,
            'stuck': total_frac * 0.15,
            'noisy': total_frac * 0.15
        }
        m, *_ = run_single_configuration(
            s_crop, size=work_size, fractions=fracs, n_flat=16, n_dark=16, dynamic_range=dyn_range, seed=600 + int(total_frac*10000)
        )
        exp6_rows.append(log_row('Exp_6', 'Доля дефектных пикселей', 'Defect_Fraction_%', round(total_frac * 100, 3), m))
    pd.DataFrame(exp6_rows).to_csv(os.path.join(out_dir, 'exp6_defect_fraction.csv'), index=False)

    # Эксперимент №7: Неравномерность источника плоского поля и межкадровый дрейф L_k
    exp7_rows = []
    for (drift, igrad, label) in [
        (0.00, 0.00, 'Идеальный источник (drift=0, grad=0)'),
        (0.05, 0.00, 'Дрейф яркости 5% (drift=0.05)'),
        (0.15, 0.00, 'Сильный дрейф 15% (drift=0.15)'),
        (0.05, 0.03, 'Дрейф 5% + градиент источника 3%'),
        (0.05, 0.08, 'Дрейф 5% + градиент источника 8%')
    ]:
        m, *_ = run_single_configuration(
            s_crop, size=work_size, drift_sigma=drift, illumination_grad=igrad, n_flat=16, n_dark=16, dynamic_range=dyn_range, seed=700 + int(igrad*100)
        )
        exp7_rows.append(log_row('Exp_7', 'Нестабильность и градиент Flat-источника', 'Source_Condition', label, m))
    pd.DataFrame(exp7_rows).to_csv(os.path.join(out_dir, 'exp7_source_nonideal.csv'), index=False)

    # Эксперимент №9: Комплексная модель на полном разрешении 1024x1024 и сравнение методов интерполяции
    exp9_rows = []
    for method in ['bilinear', 'median', 'weighted']:
        m, *_ = run_single_configuration(
            s_ref_full, size=s_ref_full.shape, sigma_g=0.02, lowfreq_amp=0.04, radial_amp=0.10,
            dark_level=15.0, dark_sigma=2.0, n_flat=16, n_dark=16, flat_level=0.60,
            drift_sigma=0.03, inpaint_method=method, dynamic_range=dyn_range, seed=900
        )
        exp9_rows.append(log_row('Exp_9', 'Комплексная модель (1024x1024, выбор интерполяции)', 'Inpaint_Method', method, m))
    pd.DataFrame(exp9_rows).to_csv(os.path.join(out_dir, 'exp9_complex_model.csv'), index=False)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(out_dir, 'all_experiments_summary.csv'), index=False)
    return summary_df
