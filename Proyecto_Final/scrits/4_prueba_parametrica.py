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
# SensorsR directory
dirsenr = os.path.join(parentDirectory, "data")
# data processing directory
makecar(parentDirectory, "ParamtOleajSensoresR")
dirparolejsenr = os.path.join(parentDirectory, "ParamtOleajSensoresR")

# sensor datasets to be calibrated
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
    
    resultados = []

    # Load data for all cases
    df_lab = [load_and_process_data(dirparolejsenr, name_lis_caso[i], tipo) for i in range(len(name_lis_caso))]
    
    # Define columns to analyze
    col_defs = [(4, 'H [m]')]
    if tipo == 'Reg':
        col_defs.append((2, 'eta_bar [m]'))

    etiquetas_caso = [name_lis_caso[i] for i in range(len(df_lab))]

    for idx_col, nom_col in col_defs:
        grupos = []
        for df in df_lab:
            col = df.iloc[:, idx_col] if isinstance(df, pd.DataFrame) else pd.Series(df[:, idx_col])
            serie = pd.to_numeric(col, errors='coerce')
            serie = serie.replace([np.inf, -np.inf], np.nan).dropna()
            grupos.append(serie.to_numpy())

        grupos_validos = [g for g in grupos if g.size > 0]
        if len(grupos_validos) < 2:
            print(f"{nom_col}: insufficient data for tests.")
            resultados.append({'tipo': tipo, 'variable': nom_col, 'test': 'ANOVA_oneway', 'status': 'insufficient'})
            resultados.append({'tipo': tipo, 'variable': nom_col, 'test': 'Welch_t', 'status': 'insufficient'})
            resultados.append({'tipo': tipo, 'variable': nom_col, 'test': 'Chi2_independence', 'status': 'insufficient'})
            continue

        # One-way ANOVA
        try:
            f_stat, p_anova = stats.f_oneway(*grupos_validos)
            print(f"One-way ANOVA {nom_col}: F={f_stat:.4g}, p={p_anova:.4g}")
            resultados.append({
                'tipo': tipo,
                'variable': nom_col,
                'test': 'ANOVA_oneway',
                'stat_F': float(f_stat),
                'p_value': float(p_anova),
                'k_grupos': int(len(grupos_validos)),
                'n_total': int(sum(g.size for g in grupos_validos))
            })
        except Exception as e:
            print(f"One-way ANOVA {nom_col}: calculation error.")
            resultados.append({
                'tipo': tipo, 'variable': nom_col, 'test': 'ANOVA_oneway',
                'status': 'error', 'message': str(e)[:200]
            })

        # Pairwise Welch t-tests (Bonferroni)
        for i, j in combinations(range(len(grupos)), 2):
            gi, gj = grupos[i], grupos[j]
            if gi.size == 0 or gj.size == 0:
                continue
            t_stat, p_val = stats.ttest_ind(gi, gj, equal_var=False, nan_policy='omit')
            p_adj = min(p_val * 3, 1.0)
            print(f"Welch t-test {nom_col} {etiquetas_caso[i]} vs {etiquetas_caso[j]}: t={t_stat:.4g}, p_adj={p_adj:.4g}")
            resultados.append({
                'tipo': tipo,
                'variable': nom_col,
                'test': 'Welch_t',
                'grupo_i': etiquetas_caso[i],
                'grupo_j': etiquetas_caso[j],
                'n_i': int(gi.size),
                'n_j': int(gj.size),
                't_stat': float(t_stat),
                'p_raw': float(p_val),
                'p_adj_bonf': float(p_adj)
            })

        # Chi-square (independence) after common binning
        try:
            pooled = np.concatenate(grupos_validos)
            if pooled.size < 2:
                print(f"Chi-square {nom_col}: insufficient data.")
                resultados.append({'tipo': tipo, 'variable': nom_col, 'test': 'Chi2_independence', 'status': 'insufficient'})
                continue
            q_edges = np.nanquantile(pooled, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            edges = np.unique(q_edges)
            if edges.size < 3:
                edges = np.linspace(np.nanmin(pooled), np.nanmax(pooled), 4)

            tabla = []
            for g in grupos:
                if g.size == 0:
                    tabla.append(np.zeros(edges.size - 1, dtype=int))
                else:
                    cnt, _ = np.histogram(g, bins=edges)
                    tabla.append(cnt)
            tabla = np.asarray(tabla)

            if tabla.sum() == 0 or tabla.shape[1] < 2:
                print(f"Chi-square {nom_col}: insufficient data.")
                resultados.append({'tipo': tipo, 'variable': nom_col, 'test': 'Chi2_independence', 'status': 'insufficient'})
            else:
                chi2, p_chi, gl, _ = stats.chi2_contingency(tabla)
                print(f"Chi-square {nom_col}: chi2={chi2:.4g}, df={gl}, p={p_chi:.4g}")
                resultados.append({
                    'tipo': tipo,
                    'variable': nom_col,
                    'test': 'Chi2_independence',
                    'chi2': float(chi2),
                    'gl': int(gl),
                    'p_value': float(p_chi),
                    'bins': int(tabla.shape[1])
                })
        except Exception as e:
            print(f"Chi-square {nom_col}: calculation error.")
            resultados.append({
                'tipo': tipo, 'variable': nom_col, 'test': 'Chi2_independence',
                'status': 'error', 'message': str(e)[:200]
            })

    # Save results to CSV
    out_csv = os.path.join(dirparolejsenr, f"resultados_parametricos{tipo}.csv")
    pd.DataFrame(resultados).to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv}")
