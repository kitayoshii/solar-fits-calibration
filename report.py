"""
report.py — Модуль этапа 8:
Генерация сводных графиков, PNG-таблиц и визуальных сравнений для презентации и защиты проекта:
- Карты истинной vs оценённой чувствительности G(x,y) + гистограмма ошибки оценки;
- Сравнение: исходный солнечный кадр S(x,y), искажённый кадр Y(x,y), восстановленный S_hat(x,y), карта остаточной ошибки;
- Графики зависимостей RMSE/PSNR/SSIM/F1 от числа кадров, уровня освещённости flat, амплитуды PRNU и доли дефектов;
- Сводная PNG-таблица «эксперимент -> параметры -> метрики».
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_gain_and_dark_comparison(gt_dict, g_hat, master_dark, bad_mask_hat, out_path='./results/plots/gain_dark_evaluation.png'):
    """
    Строит 6-панельную визуализацию:
    1) Истинная карта G_true (на рабочих пикселях), 2) Оценённая карта G_hat, 3) Гистограмма ошибки G_hat - G_true,
    4) Истинный темновой сигнал D_true, 5) Оценённый master_dark D_hat, 6) Маска дефектов (истинная vs найденная).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g_true = gt_dict['G_true']
    d_true = gt_dict['D_true']
    bad_true = gt_dict['bad_mask']
    valid = ~bad_true

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.8), dpi=130)

    vmin_g, vmax_g = np.percentile(g_true[valid], [1, 99])
    im0 = axes[0, 0].imshow(np.clip(g_true, vmin_g, vmax_g), cmap='viridis', origin='lower', vmin=vmin_g, vmax=vmax_g)
    axes[0, 0].set_title('Истинная чувствительность G_true(x,y)', fontsize=11, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(np.clip(g_hat, vmin_g, vmax_g), cmap='viridis', origin='lower', vmin=vmin_g, vmax=vmax_g)
    axes[0, 1].set_title('Оценённая карта G_hat(x,y) (Master-Flat)', fontsize=11, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    err_g = (g_hat[valid] - g_true[valid])
    axes[0, 2].hist(err_g, bins=80, color='#2b5c8f', edgecolor='black', alpha=0.85)
    axes[0, 2].axvline(0.0, color='crimson', linestyle='--', linewidth=1.5)
    title_err = f"Гистограмма ошибки (G_hat - G_true) | RMSE={np.sqrt(np.mean(err_g**2)):.5f}, MAE={np.mean(np.abs(err_g)):.5f}"
    axes[0, 2].set_title(title_err, fontsize=10.5, fontweight='bold')
    axes[0, 2].set_xlabel('Ошибка чувствительности ΔG')
    axes[0, 2].set_ylabel('Число пикселей')
    axes[0, 2].grid(True, alpha=0.3)

    vmin_d, vmax_d = np.percentile(d_true[valid], [1, 99])
    im3 = axes[1, 0].imshow(np.clip(d_true, vmin_d, vmax_d), cmap='magma', origin='lower', vmin=vmin_d, vmax=vmax_d)
    axes[1, 0].set_title('Истинный темновой сигнал D_true(x,y) [DN]', fontsize=11, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(np.clip(master_dark, vmin_d, vmax_d), cmap='magma', origin='lower', vmin=vmin_d, vmax=vmax_d)
    axes[1, 1].set_title('Оценённый Master-Dark D_hat(x,y) [DN]', fontsize=11, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    crop_s = slice(448, 576)
    overlay = np.zeros((128, 128, 3), dtype=np.float32)
    bt_c = bad_true[crop_s, crop_s]
    bh_c = bad_mask_hat[crop_s, crop_s]
    overlay[bt_c & bh_c] = [0.1, 0.85, 0.2]       # Зеленый = True Positive
    overlay[bt_c & (~bh_c)] = [0.95, 0.15, 0.15]   # Красный = False Negative
    overlay[(~bt_c) & bh_c] = [1.0, 0.75, 0.1]     # Желтый = False Positive
    axes[1, 2].imshow(overlay, origin='lower')
    axes[1, 2].set_title('Дефектные пиксели (128x128: Зел=TP, Красн=FN, Желт=FP)', fontsize=10.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_scientific_frame_comparison(s_true, y_distorted, s_hat, regions_dict, out_path='./results/comparisons/solar_correction_showcase.png'):
    """
    Строит презентационный коллаж:
    1) Эталонное солнечное изображение S(x,y) (arcsinh-шкала),
    2) Искажённое сырое изображение Y(x,y) (виньетирование, PRNU, темновой пьедестал, битые пиксели),
    3) Восстановленное изображение S_hat(x,y) после калибровки и интерполяции дефектов,
    4) Карта абсолютной ошибки |S_hat - S|,
    5) Увеличенный кроп активной области до и после коррекции,
    6) Карта сегментации солнечных зон (спокойная фотосфера, границы, яркие АО, малый контраст).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.2), dpi=130)

    vmax_s = float(np.percentile(s_true, 99.7))
    im0 = axes[0, 0].imshow(np.arcsinh(np.clip(s_true, 0, vmax_s)), cmap='inferno', origin='lower')
    axes[0, 0].set_title('1. Эталонный солнечный кадр S(x,y) — SDO/AIA 304 A (arcsinh)', fontsize=10.5, fontweight='bold')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(np.arcsinh(np.clip(y_distorted, 0, vmax_s * 1.2)), cmap='inferno', origin='lower')
    axes[0, 1].set_title('2. Искажённый кадр Y(x,y) = clip(G*S + D + N)', fontsize=10.5, fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(np.arcsinh(np.clip(s_hat, 0, vmax_s)), cmap='inferno', origin='lower')
    axes[0, 2].set_title('3. Восстановленный кадр S_hat(x,y) (Взвешенная интерполяция)', fontsize=10.5, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    abs_err = np.abs(s_hat - s_true)
    im3 = axes[1, 0].imshow(abs_err, cmap='viridis', origin='lower', vmin=0, vmax=np.percentile(abs_err, 99.0))
    err_title = f"4. Карта ошибки |S_hat - S| (RMSE={np.sqrt(np.mean(abs_err**2)):.3f} DN, MAE={np.mean(abs_err):.3f} DN)"
    axes[1, 0].set_title(err_title, fontsize=10.5, fontweight='bold')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    cy, cx = np.unravel_index(np.argmax(s_true), s_true.shape)
    r = 80
    y0, y1 = max(0, min(s_true.shape[0] - 2 * r, cy - r)), max(2 * r, min(s_true.shape[0], cy + r))
    x0, x1 = max(0, min(s_true.shape[1] - 2 * r, cx - r)), max(2 * r, min(s_true.shape[1], cx + r))
    crop_compare = np.hstack([
        np.arcsinh(np.clip(y_distorted[y0:y1, x0:x1], 0, vmax_s)),
        np.arcsinh(np.clip(s_hat[y0:y1, x0:x1], 0, vmax_s))
    ])
    im4 = axes[1, 1].imshow(crop_compare, cmap='inferno', origin='lower')
    axes[1, 1].axvline(x1 - x0 - 0.5, color='cyan', linestyle='--', linewidth=1.8)
    axes[1, 1].set_title('5. Фрагмент активной области (Слева: Y, Справа: S_hat)', fontsize=10.5, fontweight='bold')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    seg_rgb = np.zeros((s_true.shape[0], s_true.shape[1], 3), dtype=np.float32)
    seg_rgb[regions_dict['low_local_contrast']] = [0.18, 0.25, 0.45]     # Синий: малый локальный контраст
    seg_rgb[regions_dict['quiet_photosphere']] = [0.25, 0.68, 0.45]      # Зеленый: спокойная область
    seg_rgb[regions_dict['spot_boundaries']] = [0.95, 0.65, 0.15]        # Оранжевый: границы высокой крутизны
    seg_rgb[regions_dict['bright_active_regions']] = [0.92, 0.18, 0.25]  # Красный: яркие активные области
    axes[1, 2].imshow(seg_rgb, origin='lower')
    axes[1, 2].set_title('6. Зональная декомпозиция (Син:Фон, Зел:Спок, Оранж:Гран, Красн:АО)', fontsize=10.0, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def plot_experiment_curves(out_dir='./results/tables', plot_dir='./results/plots'):
    """
    Строит 4-панельный график ключевых экспериментальных зависимостей (Этапы 7 и 8).
    """
    os.makedirs(plot_dir, exist_ok=True)
    df1 = pd.read_csv(os.path.join(out_dir, 'exp1_n_flat.csv'))
    df2 = pd.read_csv(os.path.join(out_dir, 'exp2_flat_level.csv'))
    df3 = pd.read_csv(os.path.join(out_dir, 'exp3_sigma_g.csv'))
    df6 = pd.read_csv(os.path.join(out_dir, 'exp6_defect_fraction.csv'))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), dpi=130)

    # Панель 1: N_flat vs G_RMSE и PSNR
    x1 = df1['Factor_Value'].astype(int)
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    l1 = ax1.plot(x1, df1['G_RMSE'], 'o-', color='#1f77b4', linewidth=2.2, markersize=7, label='RMSE оценки G')
    l2 = ax1_twin.plot(x1, df1['PSNR'], 's--', color='#d62728', linewidth=2.2, markersize=7, label='PSNR (дБ)')
    ax1.set_title('Эксперимент №1: Влияние числа кадров плоского поля N_flat', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Число кадров плоского поля N_flat')
    ax1.set_ylabel('RMSE карты чувствительности G', color='#1f77b4')
    ax1_twin.set_ylabel('PSNR (дБ)', color='#d62728')
    ax1.set_xticks(x1)
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right')

    # Панель 2: Уровень Flat vs G_RMSE и Остаточная неравномерность
    x2 = df2['Factor_Value'].astype(float)
    ax2 = axes[0, 1]
    ax2.plot(x2, df2['G_RMSE'], 'o-', color='#2ca02c', linewidth=2.2, markersize=7, label='G_RMSE')
    ax2.plot(x2, df2['Residual_NonUniformity_MAD'], '^--', color='#ff7f0e', linewidth=2.2, markersize=7, label='Остаточная неравномерность (MAD)')
    ax2.set_title('Эксперимент №2: Влияние уровня засветки Flat-кадров (% диапазона)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Уровень яркости L_k (% динамического диапазона)')
    ax2.set_ylabel('Ошибка оценки чувствительности')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Панель 3: Амплитуда PRNU sigma_G vs Остаточная неравномерность и S_RMSE
    x3 = df3['Factor_Value'].astype(float) * 100.0
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    l31 = ax3.plot(x3, df3['G_RMSE'], 'o-', color='#9467bd', linewidth=2.2, markersize=7, label='G_RMSE после калибровки')
    l32 = ax3_twin.plot(x3, df3['S_RMSE'], 'd--', color='#8c564b', linewidth=2.2, markersize=7, label='RMSE научного кадра S_hat (DN)')
    ax3.set_title('Эксперимент №3: Устойчивость к росту амплитуды PRNU sigma_G', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Амплитуда пиксельной неоднородности sigma_G (%)')
    ax3.set_ylabel('G_RMSE', color='#9467bd')
    ax3_twin.set_ylabel('S_RMSE (DN)', color='#8c564b')
    ax3.grid(True, alpha=0.3)
    lines3 = l31 + l32
    ax3.legend(lines3, [l.get_label() for l in lines3], loc='upper left')

    # Панель 4: Доля дефектов vs Precision, Recall, F1
    x6 = df6['Factor_Value'].astype(float)
    ax6 = axes[1, 1]
    ax6.plot(x6, df6['Mask_Precision'], 'o-', color='#17becf', linewidth=2.2, markersize=7, label='Precision (Точность)')
    ax6.plot(x6, df6['Mask_Recall'], 's-', color='#bcbd22', linewidth=2.2, markersize=7, label='Recall (Полнота)')
    ax6.plot(x6, df6['Mask_F1'], '^-', color='#e377c2', linewidth=2.4, markersize=7, label='F1-мера')
    ax6.plot(x6, df6['Artifact_Free_Defects_Ratio'], 'd--', color='#333333', linewidth=1.8, markersize=6, label='Доля дефектов без артефактов')
    ax6.set_title('Эксперимент №6: Детектирование и интерполяция vs доля дефектов', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Доля дефектных пикселей матрицы (%)')
    ax6.set_ylabel('Значение метрики [0..1]')
    ax6.set_ylim(0.70, 1.03)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='lower left')

    plt.tight_layout()
    out_plot = os.path.join(plot_dir, 'experiments_curves_summary.png')
    plt.savefig(out_plot)
    plt.close(fig)
    return out_plot


def render_summary_table_png(summary_df, out_path='./results/tables/experiments_summary_table.png'):
    """
    Генерирует презентационную PNG-таблицу «Эксперимент -> Параметры -> Ключевые метрики».
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = [
        'Exp_ID', 'Varying_Factor', 'Factor_Value',
        'G_RMSE', 'G_P95Err', 'Residual_NonUniformity_STD',
        'S_RMSE', 'PSNR', 'SSIM', 'Mask_F1', 'Artifact_Free_Defects_Ratio'
    ]
    disp_df = summary_df[cols].copy()
    for c in ['G_RMSE', 'G_P95Err', 'Residual_NonUniformity_STD']:
        disp_df[c] = disp_df[c].map(lambda x: f"{x:.5f}")
    for c in ['S_RMSE', 'PSNR']:
        disp_df[c] = disp_df[c].map(lambda x: f"{x:.2f}")
    for c in ['SSIM', 'Mask_F1', 'Artifact_Free_Defects_Ratio']:
        disp_df[c] = disp_df[c].map(lambda x: f"{x:.4f}")

    headers = [
        'Эксп.', 'Фактор', 'Значение',
        'G_RMSE', 'G_P95', 'Ост.Неравн.',
        'S_RMSE', 'PSNR(дБ)', 'SSIM', 'Mask F1', 'Без артеф.'
    ]

    fig, ax = plt.subplots(figsize=(16.5, 0.42 * len(disp_df) + 1.8), dpi=130)
    ax.axis('off')
    ax.set_title('Сводная таблица экспериментов: параметры моделирования и метрики качества калибровки', fontsize=13, fontweight='bold', pad=12)

    tbl = ax.table(
        cellText=disp_df.values,
        colLabels=headers,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.32)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1b365d')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#f2f5f9')
        else:
            cell.set_facecolor('#ffffff')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path