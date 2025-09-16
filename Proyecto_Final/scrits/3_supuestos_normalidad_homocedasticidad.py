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
import re

from scipy import stats

from funciones.modulo_funciones  import (makecar,
                                         _clean_vec,
                                         _normal_test,
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
# SensorsR directory
dirsenr = os.path.join(parentDirectory, "data")
# data processing directory
makecar(parentDirectory, "ParamtOleajSensoresR")
dirparolejsenr = os.path.join(parentDirectory, "ParamtOleajSensoresR")

# names of calibrated sensor data
config = {
    'lis_caso': ['H10T12', 'H10T25', 'H10T30'],
    'caso': ['Reg']
}

name_lis_caso = {i: name for i, name in enumerate(config['lis_caso'])}
name_caso = {i: oleaje for i, oleaje in enumerate(config['caso'])}

# x sensors
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

    # Columns to use: H and eta (0-based indices)
    columnas = [(4, 'H [m]')]
    if tipo == 'Reg':
        columnas.append((2, 'mean eta [m]'))

    for icol, nombre_col in columnas:
        grupos = []
        info = []
        for i, df in enumerate(df_lab):
            if isinstance(df, pd.DataFrame) and df.shape[1] > icol:
                x = df.iloc[:, icol].values
            else:
                x = np.array([])
            x = _clean_vec(x)
            grupos.append(x)
            info.append(_normal_test(x))

        # Homoscedasticity (groups with at least 2 data points)
        grupos_validos = [g for g in grupos if g.size >= 2]
        p_lev, p_bart = np.nan, np.nan
        if len(grupos_validos) >= 2:
            try:
                p_lev = float(stats.levene(*grupos_validos, center="median").pvalue)
            except Exception:
                pass
            try:
                p_bart = float(stats.bartlett(*grupos_validos).pvalue)
            except Exception:
                pass

        # Decision
        normal_ok = all(r["ok"] for r in info if r["n"] >= 3)
        homoc_ok = (np.isnan(p_lev) or p_lev > 0.05) and (np.isnan(p_bart) or p_bart > 0.05)
        param_ok = normal_ok and homoc_ok

        # Report
        print(f"Variable: {nombre_col}")
        for j, r in enumerate(info):
            etiqueta = name_lis_caso.get(j, f"Group {j}")
            suf = f"p={r['p']:.3f}" if np.isfinite(r["p"]) else f"stat={r['stat']:.3f}"
            print(f"  {etiqueta}: n={r['n']}, {r['method']}: {'OK' if r['ok'] else 'No'} ({suf})")
        txt_lev = "NA" if np.isnan(p_lev) else f"{p_lev:.3f}"
        txt_bart = "NA" if np.isnan(p_bart) else f"{p_bart:.3f}"
        print(f"  Levene p={txt_lev} | Bartlett p={txt_bart}")
        if param_ok:
            print("  Parametric assumptions hold → use parametric tests")
        else:
            print("  Assumptions not met → use non-parametric tests")
        print("-")

        # Save results
        rows = []
        for j, r in enumerate(info):
            etiqueta = name_lis_caso.get(j, f"Group {j}")
            rows.append({
                "type": tipo,
                "variable": nombre_col,
                "group": etiqueta,
                "n": r["n"],
                "normality_test": r["method"],
                "normality_stat": r["stat"],
                "normality_p": r["p"],
                "normal_ok": r["ok"],
                "p_levene": p_lev,
                "p_bartlett": p_bart,
                "homoc_ok": homoc_ok,
                "param_ok": param_ok
            })

        df_out = pd.DataFrame(rows)
        try:
            fname = re.sub(r'[^A-Za-z0-9]+', '_', f"{tipo}_{nombre_col}").strip('_')
            out_path = os.path.join(dirparolejsenr, f"test_results_{fname}.csv")
            df_out.to_csv(out_path, index=False, float_format="%.6g")
            print(f"  Results saved to: {out_path}")
        except Exception as e:
            print(f"  Could not save results: {e}")

