# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:46 2024

@author: jrbof
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os

from scipy import stats as spstats

from funciones.modulo_funciones import (
    makecar,
    load_and_process_data,
)

# Suppress pandas FutureWarning triggered inside seaborn regarding 'use_inf_as_na'
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")

sns.set_style('darkgrid')
sns.set_style('ticks', {"axes.grid": False})

# Working directory
parentDirectory = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

# Ensure folders exist
makecar(parentDirectory, "images")
img_dir = os.path.join(parentDirectory, "images")
sensors_dir = os.path.join(parentDirectory, "data")
makecar(parentDirectory, "ParamtOleajSensoresR")
proc_dir = os.path.join(parentDirectory, "ParamtOleajSensoresR")

# Names of calibrated sensor data
config = {
    'lis_caso': ['H10T12', 'H10T25', 'H10T30'],
    'caso': ['Reg']
}

name_lis_caso = {i: name for i, name in enumerate(config['lis_caso'])}
name_caso = {i: wave for i, wave in enumerate(config['caso'])}

# Sensor x-positions
x1 = np.array([0, 1.42, 0.93, 0.84, 0.20, 0.19, 0.20, 0.195, 0.20, 0.20, 0.23, 0.20, 0.20, 0.21])
x1[2:] = np.cumsum(x1[1:-1]) + x1[2:]
x2 = x1[1:] - 0.05
x3 = x1[1:] - 0.10
x_sensors = np.hstack((x1[1:], x2, x3))

for wave_type in ['Reg']:
    print('----------------------------------------------------------------')
    print(f"Processing data --> {wave_type}")

    # Load data for all cases
    df_lab = [load_and_process_data(proc_dir, name_lis_caso[i], wave_type) for i in range(3)]

    # Columns to use: H and eta
    columns_spec = [(4, 'H [m]')]
    if wave_type == 'Reg':
        columns_spec.append((2, r'$\overline{\eta}$ [m]'))

    for col, label in columns_spec:
        print(f"-- {label} --")
        parts = []
        included_names = []
        summary_stats = []

        for i, arr in enumerate(df_lab):
            vals = arr[:, col]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                print(f"{name_lis_caso[i]} - {wave_type} | no valid data")
                continue

            mean = float(np.mean(vals))
            median = float(np.median(vals))
            variance = float(np.var(vals, ddof=1)) if vals.size > 1 else np.nan
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            if vals.size > 0:
                q75, q25 = np.percentile(vals, [75, 25])
                iqr = float(q75 - q25)
            else:
                iqr = np.nan
            cv = float(std / abs(mean)) if np.isfinite(std) and mean != 0 else np.nan

            print(
                f"{name_lis_caso[i]} - {wave_type} | "
                f"mean: {mean:.4f} | median: {median:.4f} | variance: {variance:.4f} | "
                f"std: {std:.4f} | IQR: {iqr:.4f} | CV: {cv:.4f}"
            )

            parts.append(pd.DataFrame({"value": vals, "case": name_lis_caso[i]}))
            included_names.append(name_lis_caso[i])

            summary_stats.append({
                "case": name_lis_caso[i],
                "type": wave_type,
                "column": col,
                "n": int(vals.size),
                "mean": mean,
                "median": median,
                "variance": float(variance) if np.isfinite(variance) else np.nan,
                "std": float(std) if np.isfinite(std) else np.nan,
                "IQR": float(iqr) if np.isfinite(iqr) else np.nan,
                "CV": float(cv) if np.isfinite(cv) else np.nan,
            })

        # Save/update CSV with the summary (if data available)
        if summary_stats:
            summary_path = os.path.join(proc_dir, f"summary_statistics_{wave_type}_col{col}.csv")
            pd.DataFrame(summary_stats).to_csv(summary_path, index=False, float_format="%.6f")

        if not parts:
            print(f"{wave_type} | col {col} no data to plot")
            continue

        df_plot = pd.concat(parts, ignore_index=True)

        # Colors: one color per case
        colors = ['slategray', 'purple', 'indigo']
        palette_map = {case: colors[i % len(colors)] for i, case in enumerate(included_names)}
        palette_ordered = [palette_map[case] for case in included_names]

        # Histogram + KDE faceted by case with unique color per case
        g = sns.FacetGrid(
            data=df_plot,
            col="case",
            col_order=included_names,
            sharex=False, sharey=False,
            height=4, aspect=1.1
        )
        for ax, case in zip(g.axes.flat, included_names):
            c = palette_map[case]
            subset = df_plot[df_plot["case"] == case]
            sns.histplot(subset, x="value", stat="density", bins="auto", color=c, alpha=0.6, ax=ax)
            sns.kdeplot(data=subset, x="value", color=c, ax=ax)

        g.set_axis_labels(label, "Density")
        g.figure.suptitle(f"Distributions \n {wave_type}")
        g.figure.tight_layout()
        g.figure.subplots_adjust(top=0.85)

        img_path = os.path.join(img_dir, f"distributions_{wave_type}_{col}.png")
        g.figure.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Overlaid KDE by case (single plot)
        fig, ax = plt.subplots(figsize=(6, 4))
        for case in included_names:
            subset = df_plot[df_plot["case"] == case]
            sns.kdeplot(data=subset, x="value", ax=ax, label=case, color=palette_map[case])
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"KDE - {wave_type}")
        ax.legend(title="Cases")
        fig.tight_layout()

        img_path_kde = os.path.join(img_dir, f"kde_{wave_type}_{col}.png")
        fig.savefig(img_path_kde, dpi=300, bbox_inches="tight")
        plt.close()

        # QQ-plots by case to check normality
        ncols = len(included_names)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        for ax, case in zip(axes, included_names):
            subset_vals = df_plot.loc[df_plot["case"] == case, "value"].values
            spstats.probplot(subset_vals, dist="norm", plot=ax)
            ax.set_title(case)
            ax.grid(False)

        fig.suptitle(f"QQ-plots Normality - {wave_type} | {label}")
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        img_path_qq = os.path.join(img_dir, f"qq_plots_{wave_type}_{col}.png")

        fig.savefig(img_path_qq, dpi=300, bbox_inches="tight")
        plt.close()

        # Boxplots by case (horizontal) with unique color per case
        g2 = sns.catplot(
            data=df_plot,
            x="value",
            y="case",
            order=included_names,
            kind="box",
            palette=palette_ordered,
            height=4, aspect=1.4
        )
        g2.set_axis_labels(label, "")
        g2.figure.suptitle(f"Boxplots \n {wave_type}")
        g2.figure.tight_layout()
        g2.figure.subplots_adjust(top=0.88)

        img_path_box = os.path.join(img_dir, f"boxplots_{wave_type}_{col}.png")
        g2.figure.savefig(img_path_box, dpi=300, bbox_inches="tight")

        plt.close()
