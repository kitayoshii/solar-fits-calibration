"""
build_project_bundle.py — Скрипт автоматической сборки всего проекта под ключ:
1. Проверяет структуру папок и копирует исходные FITS SDO/AIA 304 A в ./data/solar/.
2. Создает requirements.txt и README.md для группы.
3. Запускает вычислительный пайплайн (этапы 1–8) и сохраняет все FITS, CSV и PNG.
4. Генерирует оформленный отчет для защиты (Solar_FITS_Calibration_Report.docx).
5. Генерирует презентацию для защиты 12 числа (Solar_FITS_Calibration_Defense.pptx).
6. Упаковывает все файлы проекта, данные, результаты, отчет и презентацию в Solar_FITS_Calibration_Project.zip.
"""
import os
import glob
import shutil
import zipfile

# 1. Подготовка директорий и копирование FITS
DIRS = [
    'data/solar',
    'data/normalized',
    'data/ground_truth',
    'data/calibration/dark',
    'data/calibration/flat',
    'data/calibration/masters',
    'data/corrected',
    'results/previews',
    'results/tables',
    'results/plots',
    'results/comparisons',
    'defense_materials'
]
for d in DIRS:
    os.makedirs(d, exist_ok=True)

for fits_file in sorted(glob.glob('AIA*.fits')):
    target = os.path.join('data/solar', os.path.basename(fits_file))
    if not os.path.exists(target):
        shutil.copy(fits_file, target)

# 2. Автоматическая нормализация переносов строк в report.py (если скачана ранняя версия)
if os.path.exists('report.py'):
    with open('report.py', 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    fixed = []
    i = 0
    while i < len(lines):
        if 'set_title(' in lines[i] and not lines[i].rstrip().endswith(')'):
            fixed.append(lines[i] + ' | ' + lines[i + 1].lstrip())
            i += 2
        else:
            fixed.append(lines[i])
            i += 1
    with open('report.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed) + '\n')

# 3. Запись requirements.txt
REQUIREMENTS_TXT = """numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
scikit-image>=0.20.0
astropy>=5.3.0
python-docx>=1.1.0
python-pptx>=0.6.21
"""
with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(REQUIREMENTS_TXT)

# 4. Запись README.md для группы
README_MD = """# Калибровка астрономических солнечных FITS-изображений (SDO/AIA 304 Å)

Готовое модульное Python-приложение для моделирования фотоприёмника (Flat-Field, Dark-Frame, PRNU, виньетирование, маска дефектов `dead`/`hot`/`stuck`/`noisy`), робастной калибровки по статистикам медианы и MAD и восстановления реальных солнечных FITS-кадров.

## Быстрый старт (для ребят из группы)

1. **Откройте папку проекта в PyCharm** (`File -> Open`).
2. **Установите библиотеки** в терминале PyCharm (`Alt + F12`):
   ```bash
   pip install -r requirements.txt"""