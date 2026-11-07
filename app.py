
"""
app.py — Графическое приложение для калибровки SDO/AIA FITS.

Возможности:
- Просмотр FITS-файлов.
- Просмотр PNG-изображений и графиков.
- Запуск полного расчётного пайплайна.
- Просмотр результатов экспериментов.
- Просмотр коллажа коррекции.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from fits_utils import read_fits
from run_pipeline import run_project


class SolarFitsApp:

    def __init__(self, root):
        self.root = root

        self.root.title(
            "Астрономический калибратор SDO/AIA FITS — Защита проекта"
        )
        self.root.geometry("1150x750")
        self.root.minsize(850, 550)

        # -------------------------------
        # Верхняя панель управления
        # -------------------------------

        control_frame = ttk.LabelFrame(
            root,
            text=" Управление проектом ",
            padding=10
        )
        control_frame.pack(
            side=tk.TOP,
            fill=tk.X,
            padx=10,
            pady=5
        )

        ttk.Button(
            control_frame,
            text="▶ Запустить пайплайн (Все этапы)",
            command=self.run_pipeline_gui
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="📂 Открыть FITS файл...",
            command=self.load_fits_dialog
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="📊 Показать графики экспериментов",
            command=self.show_experiments_plot
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="📋 Показать коллаж коррекции",
            command=self.show_correction_collaqe
        ).pack(side=tk.LEFT, padx=5)

        # -------------------------------
        # Основная рабочая область
        # -------------------------------

        container = ttk.Frame(root)

        container.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=5
        )

        # Левая панель: дерево файлов

        left_frame = ttk.LabelFrame(
            container,
            text=" Файлы FITS и Результаты ",
            padding=5
        )

        left_frame.pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=5
        )

        self.tree = ttk.Treeview(
            left_frame,
            show="tree",
            selectmode="browse",
            height=25
        )

        self.tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True
        )

        # Полоса прокрутки дерева

        tree_scroll = ttk.Scrollbar(
            left_frame,
            orient=tk.VERTICAL,
            command=self.tree.yview
        )

        tree_scroll.pack(
            side=tk.RIGHT,
            fill=tk.Y
        )

        self.tree.configure(
            yscrollcommand=tree_scroll.set
        )

        self.tree.bind(
            "<Double-1>",
            self.on_item_double_click
        )

        # Правая панель: визуализация

        self.right_frame = ttk.LabelFrame(
            container,
            text=" Просмотр изображения / Графика ",
            padding=5
        )

        self.right_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=5
        )

        # -------------------------------
        # Matplotlib
        # -------------------------------

        self.fig, self.ax = plt.subplots(
            figsize=(7, 6),
            dpi=100
        )

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.right_frame
        )

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        self.ax.set_title("Окно визуализации SDO/AIA FITS")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.canvas.draw()

        # -------------------------------
        # Строка состояния
        # -------------------------------

        self.status_var = tk.StringVar(
            value="Готово к работе. Выберите файл слева или запустите пайплайн."
        )

        status_bar = ttk.Label(
            root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=5
        )

        status_bar.pack(
            side=tk.BOTTOM,
            fill=tk.X
        )

        # Первоначальное заполнение дерева файлов

        self.refresh_file_tree()

        # Обработка закрытия окна

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close
        )

    # ==================================================
    # ОБНОВЛЕНИЕ ДЕРЕВА ФАЙЛОВ
    # ==================================================

    def refresh_file_tree(self):

        for item in self.tree.get_children():
            self.tree.delete(item)

        folders = [
            "./data/solar",
            "./data/corrected",
            "./results/plots",
            "./results/comparisons",
            "./results/tables"
        ]

        for folder in folders:

            if not os.path.isdir(folder):
                continue

            folder_node = self.tree.insert(
                "",
                "end",
                text=folder,
                open=True
            )

            try:
                files = sorted(os.listdir(folder))

                for filename in files:

                    filepath = os.path.join(
                        folder,
                        filename
                    )

                    if os.path.isfile(filepath):

                        self.tree.insert(
                            folder_node,
                            "end",
                            text=filename,
                            values=(filepath,)
                        )

            except OSError as error:
                print(
                    f"Не удалось прочитать папку {folder}: {error}"
                )

    # ==================================================
    # ОБРАБОТКА ДВОЙНОГО ЩЕЛЧКА ПО ФАЙЛУ
    # ==================================================

    def on_item_double_click(self, event):

        selection = self.tree.selection()

        if not selection:
            return

        item = self.tree.item(selection[0])

        values = item.get("values", [])

        if not values:
            return

        filepath = values[0]

        if not os.path.isfile(filepath):
            return

        extension = os.path.splitext(filepath)[1].lower()

        if extension in (".fits", ".fit", ".fts"):

            self.plot_fits_file(filepath)

        elif extension in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):

            self.plot_png_file(filepath)

    # ==================================================
    # ОЧИСТКА ОБЛАСТИ ВИЗУАЛИЗАЦИИ
    # ==================================================

    def clear_plot(self):

        self.fig.clear()

        self.ax = self.fig.add_subplot(111)

    # ==================================================
    # ОТОБРАЖЕНИЕ FITS
    # ==================================================

    def plot_fits_file(self, filepath):

        try:

            data, hdr = read_fits(filepath)

            if data is None:
                raise ValueError("FITS-файл не содержит данных.")

            data = np.asarray(data)

            if data.ndim != 2:
                raise ValueError(
                    f"Ожидалось двумерное изображение, получена форма: {data.shape}"
                )

            if not np.any(np.isfinite(data)):
                raise ValueError(
                    "Изображение не содержит конечных числовых значений."
                )

            self.clear_plot()

            # Ограничение яркости для корректного отображения

            valid_data = data[np.isfinite(data)]

            vmin = np.percentile(valid_data, 0.5)
            vmax = np.percentile(valid_data, 99.5)

            if vmax <= vmin:
                vmax = vmin + 1

            # Логарифмически-подобное отображение

            clipped = np.clip(data, max(vmin, 0), vmax)

            image = self.ax.imshow(
                np.arcsinh(clipped),
                cmap="inferno",
                origin="lower",
                interpolation="nearest"
            )

            self.ax.set_title(
                f"FITS: {os.path.basename(filepath)}\n"
                f"Размер: {data.shape} | "
                f"EXPTIME: {hdr.get('EXPTIME', 'N/A')} s",
                fontsize=10
            )

            self.ax.set_xlabel("X, пиксели")
            self.ax.set_ylabel("Y, пиксели")

            self.fig.colorbar(
                image,
                ax=self.ax,
                fraction=0.046,
                pad=0.04,
                label="Интенсивность (arcsinh)"
            )

            self.fig.tight_layout()

            self.canvas.draw()

            self.status_var.set(
                f"Просмотр FITS: {filepath}"
            )

        except Exception as error:

            messagebox.showerror(
                "Ошибка чтения FITS",
                str(error)
            )

    # ==================================================
    # ОТОБРАЖЕНИЕ PNG И ДРУГИХ ИЗОБРАЖЕНИЙ
    # ==================================================

    def plot_png_file(self, filepath):

        try:

            img = plt.imread(filepath)

            self.clear_plot()

            self.ax.imshow(img)

            self.ax.set_title(
                f"График / Превью: {os.path.basename(filepath)}",
                fontsize=10
            )

            self.ax.axis("off")

            self.fig.tight_layout()

            self.canvas.draw()

            self.status_var.set(
                f"Просмотр изображения: {filepath}"
            )

        except Exception as error:

            messagebox.showerror(
                "Ошибка открытия изображения",
                str(error)
            )

    # ==================================================
    # ОТКРЫТИЕ FITS ЧЕРЕЗ ДИАЛОГ
    # ==================================================

    def load_fits_dialog(self):

        filepath = filedialog.askopenfilename(
            title="Выберите FITS-файл",
            filetypes=[
                ("FITS files", "*.fits *.fit *.fts"),
                ("All files", "*.*")
            ]
        )

        if filepath:
            self.plot_fits_file(filepath)

    # ==================================================
    # ГРАФИКИ ЭКСПЕРИМЕНТОВ
    # ==================================================

    def show_experiments_plot(self):

        filepath = (
            "./results/plots/experiments_curves_summary.png"
        )

        if os.path.exists(filepath):

            self.plot_png_file(filepath)

        else:

            messagebox.showwarning(
                "Нет файла",
                "Сначала запустите пайплайн (--mode all), "
                "чтобы сгенерировать графики экспериментов!"
            )

    # ==================================================
    # КОЛЛАЖ КОРРЕКЦИИ
    # ==================================================

    def show_correction_collaqe(self):

        filepath = (
            "./results/comparisons/solar_correction_showcase.png"
        )

        if os.path.exists(filepath):

            self.plot_png_file(filepath)

        else:

            messagebox.showwarning(
                "Нет файла",
                "Сначала запустите пайплайн для генерации "
                "коллажа коррекции!"
            )

    # ==================================================
    # ЗАПУСК ПОЛНОГО РАСЧЁТНОГО ПАЙПЛАЙНА
    # ==================================================

    def run_pipeline_gui(self):

        self.status_var.set(
            "Выполняется расчёт пайплайна (этапы 1–8)... "
            "Пожалуйста, подождите."
        )

        self.root.update_idletasks()

        try:

            run_project(
                mode="all",
                work_size=(256, 256)
            )

            self.refresh_file_tree()

            self.status_var.set(
                "Пайплайн успешно завершён! Все артефакты обновлены."
            )

            messagebox.showinfo(
                "Успех",
                "Все этапы калибровки и эксперименты успешно выполнены!"
            )

        except Exception as error:

            self.status_var.set(
                "Ошибка при выполнении пайплайна."
            )

            messagebox.showerror(
                "Ошибка выполнения пайплайна",
                str(error)
            )

    # ==================================================
    # ЗАКРЫТИЕ ПРИЛОЖЕНИЯ
    # ==================================================

    def on_close(self):

        try:
            plt.close(self.fig)
        except Exception:
            pass

        self.root.destroy()


# ======================================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ======================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = SolarFitsApp(root)

    root.mainloop()
