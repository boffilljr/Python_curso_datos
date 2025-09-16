# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:46 2024

@author: jrbof
"""

import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os

from itertools import combinations
from scipy import stats


from funciones.modulo_funciones  import (makecar,
                                         load_and_process_data,
                                         )

# Suppress pandas FutureWarning triggered inside seaborn regarding 'use_inf_as_na'
warnings.filterwarnings("ignore", message=".*use_inf_as_na option is deprecated.*")

sns.set_style('darkgrid')
sns.set_style('ticks', {"axes.grid":False})
# working directory
parentDirectory = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

makecar(parentDirectory, "images")
# directory to save images
dirimg = os.path.join(parentDirectory, "images")
# SensoresR directory
dirsenr = os.path.join(parentDirectory, "data")
# data processing directory
makecar(parentDirectory, "ParamtOleajSensoresR")
dirparolejsenr = os.path.join(parentDirectory, "ParamtOleajSensoresR")

# calibrated sensor data names
config = {
    'lis_caso': ['H10T12', 'H10T25', 'H10T30'],
    'caso': ['Reg']
}

name_lis_caso = {i: name for i, name in enumerate(config['lis_caso'])}
name_caso = {i: oleaje for i, oleaje in enumerate(config['caso'])}

# sensor x positions
x1 = np.array([0, 1.42, 0.93, 0.84, 0.20, 0.19, 0.20, 0.195, 0.20, 0.20, 0.23, 0.20, 0.20, 0.21])

x1[2:] = np.cumsum(x1[1:-1]) + x1[2:]
x2 = x1[1:] - 0.05
x3 = x1[1:] - 0.10

x_sensores = np.hstack((x1[1:], x2, x3))


for tipo in ['Reg']:
    print('----------------------------------------------------------------')
    print(f"Processing data --> {tipo}")
    
    # Load data for all cases
    df_lab = [load_and_process_data(dirparolejsenr, name_lis_caso[i], tipo) for i in range(3)]
    
    # Define columns to analyze
    col_defs = [(4, 'H [m]')]
    if tipo == 'Reg':
        col_defs.append((2, 'eta_bar [m]'))

# Nonparametric tests and save results
for col_idx, col_label in col_defs:
    try:
        results = []
        series_list = []
        names = []
        for i, df in enumerate(df_lab):
            if col_idx >= df.shape[1]:
                continue
            s = df.iloc[:, col_idx] if isinstance(df, pd.DataFrame) else pd.Series(df[:, col_idx])
            series_list.append(s)
            names.append(name_lis_caso.get(i, f"Group {i+1}"))

        k = len(series_list)
        if k < 2:
            print(f"[{col_label}] Insufficient data for nonparametric tests.")
            # Save informative empty file
            safe_label = (
                col_label.replace('[', '').replace(']', '').replace(' ', '_').replace('/', '-')
            )
            out_path = os.path.join(
                dirparolejsenr, f"nonparametric_results_{tipo}_{safe_label}.csv"
            )
            pd.DataFrame([], columns=[
                'variable', 'design', 'k', 'test', 'comparison', 'statistic', 'p_value', 'p_adjusted'
            ]).to_csv(out_path, index=False)
            continue

        # Determine if paired design (exactly the same indices in all groups)
        paired = all(series_list[0].index.equals(s.index) for s in series_list[1:])

        print(f"==> {col_label}: {('paired' if paired else 'independent')} (k={k})")

        # Global test
        if k == 2:
            if paired:
                mat = pd.concat(series_list, axis=1).dropna().iloc[:, 0:2]
                a, b = mat.iloc[:, 0].values, mat.iloc[:, 1].values
                if len(a) == 0:
                    print("Wilcoxon: insufficient data.")
                else:
                    stat, p = stats.wilcoxon(a, b, alternative='two-sided')
                    print(f"Wilcoxon: statistic={stat:.4f}, p={p:.4g}")
                    results.append({
                        'variable': col_label,
                        'design': 'paired',
                        'k': k,
                        'test': 'Wilcoxon',
                        'comparison': 'global',
                        'statistic': float(stat),
                        'p_value': float(p),
                        'p_adjusted': None
                    })
            else:
                a = series_list[0].dropna().values
                b = series_list[1].dropna().values
                if len(a) == 0 or len(b) == 0:
                    print("Mann–Whitney: insufficient data.")
                else:
                    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                    print(f"Mann–Whitney U: statistic={stat:.4f}, p={p:.4g}")
                    results.append({
                        'variable': col_label,
                        'design': 'independent',
                        'k': k,
                        'test': 'Mann–Whitney U',
                        'comparison': f'{names[0]} vs {names[1]} (global)',
                        'statistic': float(stat),
                        'p_value': float(p),
                        'p_adjusted': None
                    })
        else:
            if paired:
                mat = pd.concat(series_list, axis=1).dropna()
                if mat.shape[0] == 0:
                    print("Friedman: insufficient data.")
                else:
                    stat, p = stats.friedmanchisquare(*[mat.iloc[:, i].values for i in range(k)])
                    print(f"Friedman: statistic={stat:.4f}, p={p:.4g}")
                    results.append({
                        'variable': col_label,
                        'design': 'paired',
                        'k': k,
                        'test': 'Friedman',
                        'comparison': 'global',
                        'statistic': float(stat),
                        'p_value': float(p),
                        'p_adjusted': None
                    })
            else:
                arrays = [s.dropna().values for s in series_list]
                if any(len(a) == 0 for a in arrays):
                    print("Kruskal–Wallis: insufficient data.")
                else:
                    stat, p = stats.kruskal(*arrays)
                    print(f"Kruskal–Wallis: statistic={stat:.4f}, p={p:.4g}")
                    results.append({
                        'variable': col_label,
                        'design': 'independent',
                        'k': k,
                        'test': 'Kruskal–Wallis',
                        'comparison': 'global',
                        'statistic': float(stat),
                        'p_value': float(p),
                        'p_adjusted': None
                    })

            # Post-hoc comparisons with Bonferroni correction
            comps = list(combinations(range(k), 2))
            m = len(comps)
            for i, j in comps:
                if paired:
                    pair = pd.concat([series_list[i], series_list[j]], axis=1).dropna()
                    if pair.shape[0] == 0:
                        continue
                    stat_ij, p_ij = stats.wilcoxon(
                        pair.iloc[:, 0].values, pair.iloc[:, 1].values, alternative='two-sided'
                    )
                    p_adj = min(1.0, p_ij * m)
                    print(f"Post-hoc Wilcoxon ({names[i]} vs {names[j]}): W={stat_ij:.4f}, p_adj={p_adj:.4g}")
                    results.append({
                        'variable': col_label,
                        'design': 'paired',
                        'k': k,
                        'test': 'Wilcoxon (post-hoc)',
                        'comparison': f'{names[i]} vs {names[j]}',
                        'statistic': float(stat_ij),
                        'p_value': float(p_ij),
                        'p_adjusted': float(p_adj)
                    })
                else:
                    a = series_list[i].dropna().values
                    b = series_list[j].dropna().values
                    if len(a) == 0 or len(b) == 0:
                        continue
                    stat_ij, p_ij = stats.mannwhitneyu(a, b, alternative='two-sided')
                    p_adj = min(1.0, p_ij * m)
                    print(f"Post-hoc Mann–Whitney ({names[i]} vs {names[j]}): U={stat_ij:.4f}, p_adj={p_adj:.4g}")
                    results.append({
                        'variable': col_label,
                        'design': 'independent',
                        'k': k,
                        'test': 'Mann–Whitney U (post-hoc)',
                        'comparison': f'{names[i]} vs {names[j]}',
                        'statistic': float(stat_ij),
                        'p_value': float(p_ij),
                        'p_adjusted': float(p_adj)
                    })

        # Save results to CSV per variable
        safe_label = (
            col_label.replace('[', '').replace(']', '').replace(' ', '_').replace('/', '-')
        )
        out_path = os.path.join(
            dirparolejsenr, f"nonparametric_results_{tipo}_{safe_label}.csv"
        )
        df_results = pd.DataFrame(results, columns=[
            'variable', 'design', 'k', 'test', 'comparison', 'statistic', 'p_value', 'p_adjusted'
        ])
        df_results.to_csv(out_path, index=False)
        print(f"Results saved to: {out_path}")

    except Exception as e:
        print(f"Error in nonparametric tests for {col_label}: {e}")
