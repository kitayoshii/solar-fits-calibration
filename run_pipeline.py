"""
run_pipeline.py — Главный управляющий скрипт приложения (CLI).
Позволяет запускать весь пайплайн калибровки и экспериментов одной командой
или выбирать отдельные режимы через аргументы командной строки.

Примеры запуска в PyCharm / терминале:
  python run_pipeline.py --mode all          # Полный прогон всех этапов 1–8
  python run_pipeline.py --mode quick        # Быстрый прогон этапов 1–6 и графиков (без долгих таблиц этапа 7)
  python run_pipeline.py --mode stage1       # Только инспекция солнечных FITS и генерация превью
"""
import os
import argparse
import numpy as np
import pandas as pd

from generator import (
    inspect_and_prepare_solar_fits,
    build_ground_truth_detector,
    generate_dark_series,
    generate_flat_series,
    simulate_frame
)
from calibration import build_master_dark, build_master_flat
from detection import detect_bad_pixels
from correction import correct_scientific_frame
from metrics import (
    evaluate_gain_metrics,
    evaluate_mask_metrics,
    evaluate_image_metrics,
    evaluate_regional_metrics
)
from experiments import run_all_minimum_experiments
from report import (
    plot_gain_and_dark_comparison,
    plot_scientific_frame_comparison,
    plot_experiment_curves,
    render_summary_table_png
)


def run_project(mode='all', work_size=(256, 256)):
    print("=== ЭТАП 1: Загрузка и подготовка реальных солнечных FITS ===")
    df_solar, solar_frames = inspect_and_prepare_solar_fits(
        input_dir='./data/solar',
        norm_dir='./data/normalized',
        preview_dir='./results/previews',
        summary_csv='./results/tables/stage1_fits_summary.csv'
    )
    print(f"Обработано кадров: {len(solar_frames)}. Сводка сохранена в ./results/tables/stage1_fits_summary.csv")
    if mode == 'stage1':
        return

    # Выбираем кадр с максимальным динамическим диапазоном (AIA20261001_0608_0304.fits)
    ref_idx = min(3, len(solar_frames) - 1)
    ref_fname, s_raw, s_norm01, ref_hdr = solar_frames[ref_idx]
    s_ref = np.clip(s_raw - np.min(s_raw), 0.0, None)
    dyn_range = float(np.max(s_ref) * 1.25)
    size = s_ref.shape

    print("\n=== ЭТАП 2: Генерация истинных карт фотоприемника (Ground Truth) ===")
    fractions_base = {'dead': 0.0005, 'hot': 0.0005, 'stuck': 0.0002, 'noisy': 0.0003}
    gt_dict = build_ground_truth_detector(
        size=size,
        sigma_g=0.02,
        lowfreq_amp=0.04,
        lowfreq_scale=8,
        radial_amp=0.10,
        dark_level=15.0,
        dark_sigma=2.0,
        fractions=fractions_base,
        dynamic_range=dyn_range,
        seed=42,
        save_npz_path='./data/ground_truth/detector_gt.npz',
        save_json_path='./data/ground_truth/generation_params.json'
    )

    print("\n=== ЭТАП 3 и 4: Генерация серий Flat/Dark и построение калибровочных карт ===")
    dark_32 = generate_dark_series(
        n_frames=32,
        gt_dict=gt_dict,
        read_noise=2.0,
        seed=100,
        save_dir='./data/calibration/dark/N32',
        exptime=float(ref_hdr.get('EXPTIME', 2.90083))
    )

    stage4_records = []
    flat_levels = [0.20, 0.40, 0.60, 0.80]
    series_lengths = [1, 4, 8, 16, 32]

    best_master_dark = None
    best_g_hat = None
    best_bad_mask = None

    for lvl in flat_levels:
        save_f_dir = f"./data/calibration/flat/L{int(lvl*100):02d}_N16" if int(lvl*100) == 60 else None
        flat_32 = generate_flat_series(
            n_frames=32,
            level_fraction=lvl,
            gt_dict=gt_dict,
            read_noise=2.0,
            gain_e_per_dn=2.0,
            drift_sigma=0.0,
            seed=200 + int(lvl * 100),
            save_dir=save_f_dir,
            exptime=float(ref_hdr.get('EXPTIME', 2.90083))
        )
        for n_f in series_lengths:
            d_sub = dark_32[:n_f]
            f_sub = flat_32[:n_f]
            m_dark, _ = build_master_dark(d_sub)
            g_hat, _, _ = build_master_flat(f_sub, m_dark)
            b_mask, b_types, b_counts = detect_bad_pixels(g_hat, m_dark, dark_stack=d_sub, flat_stack=f_sub)

            g_metrics = evaluate_gain_metrics(gt_dict['G_true'], g_hat, gt_dict['bad_mask'])
            m_metrics = evaluate_mask_metrics(gt_dict['bad_mask'], b_mask, gt_dict['defect_types'])

            rec = {
                'Flat_Level_%': int(lvl * 100),
                'N_Frames': n_f,
                'G_RMSE': round(g_metrics['G_RMSE'], 5),
                'G_MAE': round(g_metrics['G_MAE'], 5),
                'Mask_Precision': round(m_metrics['Mask_Precision'], 4),
                'Mask_Recall': round(m_metrics['Mask_Recall'], 4),
                'Mask_F1': round(m_metrics['Mask_F1'], 4),
                'Found_Dead': b_counts['dead'],
                'Found_Hot': b_counts['hot'],
                'Found_Stuck': b_counts['stuck'],
                'Found_Noisy': b_counts['noisy'],
                'Found_Total': b_counts['total_bad']
            }
            stage4_records.append(rec)

            if int(lvl * 100) == 60 and n_f == 16:
                best_master_dark = m_dark
                best_g_hat = g_hat
                best_bad_mask = b_mask
                os.makedirs('./data/calibration/masters', exist_ok=True)
                np.savez_compressed(
                    './data/calibration/masters/master_calib_L60_N16.npz',
                    master_dark=m_dark, g_hat=g_hat, bad_mask_hat=b_mask, defect_types_hat=b_types
                )

    df_stage4 = pd.DataFrame(stage4_records)
    df_stage4.to_csv('./results/tables/stage4_calibration_grid.csv', index=False)

    print("\n=== ЭТАП 5 и 6: Искажение солнечных кадров, сравнение интерполяций и зональная оценка ===")
    rng_sci = np.random.default_rng(999)
    y_distorted = simulate_frame(s_ref, gt_dict, read_noise=2.0, gain_e_per_dn=2.0, rng=rng_sci)

    interp_records = []
    s_hat_best = None
    for method in ['bilinear', 'median', 'weighted']:
        out_fits = f"./data/corrected/{ref_fname.replace('.fits', f'_corr_{method}.fits')}"
        s_hat_m, _ = correct_scientific_frame(
            y_distorted,
            best_master_dark,
            best_g_hat,
            best_bad_mask,
            method=method,
            clip_range=(0.0, dyn_range),
            save_fits_path=out_fits,
            orig_header=ref_hdr,
            calib_info={'FLATLVL': 60, 'NFLAT': 16, 'NDARK': 16}
        )
        im_m = evaluate_image_metrics(s_ref, s_hat_m, gt_dict['bad_mask'], dynamic_range=dyn_range)
        im_m['Method'] = method
        interp_records.append(im_m)
        if method == 'weighted':
            s_hat_best = s_hat_m

    df_interp = pd.DataFrame(interp_records)
    df_interp.to_csv('./results/tables/stage5_inpainting_comparison.csv', index=False)

    reg_metrics, regions_dict = evaluate_regional_metrics(s_ref, s_hat_best)
    df_reg = pd.DataFrame.from_dict(reg_metrics, orient='index').reset_index().rename(columns={'index': 'Region'})
    df_reg.to_csv('./results/tables/stage6_regional_evaluation.csv', index=False)

    plot_gain_and_dark_comparison(gt_dict, best_g_hat, best_master_dark, best_bad_mask)
    plot_scientific_frame_comparison(s_ref, y_distorted, s_hat_best, regions_dict)

    if mode == 'all':
        print("\n=== ЭТАП 7: Запуск минимального набора экспериментов (№1–7 и №9) ===")
        summary_df = run_all_minimum_experiments(s_ref, out_dir='./results/tables', work_size=work_size)
        print("\n=== ЭТАП 8: Генерация графиков и сводных таблиц презентации ===")
        plot_experiment_curves('./results/tables', './results/plots')
        render_summary_table_png(summary_df, './results/tables/experiments_summary_table.png')

    print("\nГотово! Все выходные файлы сохранены в ./data/ и ./results/.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Калибровка астрономических солнечных FITS-изображений (SDO/AIA)')
    parser.add_argument('--mode', choices=['all', 'quick', 'stage1'], default='all', help='Режим запуска пайплайна')
    parser.add_argument('--crop', type=int, default=512, help='Размер рабочей области для серийных экспериментов этапа 7')
    args = parser.parse_args()
    run_project(mode=args.mode, work_size=(args.crop, args.crop))
