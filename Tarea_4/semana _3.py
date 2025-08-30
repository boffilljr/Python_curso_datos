# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:48:36 2025

@author: jrbof
"""

import matplotlib.pyplot as plt
import matplotlib as mpl 
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import os

from scipy.stats import pearsonr

#directorio de trabajo
wdir = os.path.dirname(os.path.abspath(__file__))
#directorio de datos de trabajo
path_data = os.path.join(wdir, 'data')

#crear carpeta 'imagenes' si no existe
path_imagenes = os.path.join(wdir, 'imagenes')
os.makedirs(path_imagenes, exist_ok=True)

path_resultados = os.path.join(wdir, 'resultados')
os.makedirs(path_resultados, exist_ok=True)


#lectura de dataframes
csv_files = sorted(glob.glob(os.path.join(path_data, '*.csv')))

if not csv_files:
    raise FileNotFoundError(f'No se encontraron archivos .csv en {path_data}')

#trabajo con Terminos_lagoon_TA_DIC_2023_RawData.csv
csv_path = csv_files[0]
df = pd.read_csv(csv_path, sep=',', encoding='utf-8')

'''
#Regresión_lineal
'''
#regresiones lineales (seaborn): Salinidad vs Temperatura / DIC vs Salinidad
#preparación de datos numéricos
df_reg = df.loc[:, ['sal_psu', 'temp_c', 'dic_micromol_kg', 'season']].copy()
for col in ['sal_psu', 'temp_c', 'dic_micromol_kg']:
    df_reg[col] = pd.to_numeric(df_reg[col], errors='coerce')

#salinidad (x) vs Temperatura (y) con lineas por temporada
sub_st = df_reg.dropna(subset=['sal_psu', 'temp_c'])
g1 = sns.lmplot(
    data=sub_st,
    x='sal_psu', y='temp_c',
    hue='season',
    height=5, aspect=1.3,
    ci=95,
    scatter_kws=dict(alpha=0.5, s=60, edgecolor='none')
)
g1.set_axis_labels('Salinidad (PSU)', 'Temperatura (°C)')
g1.ax.set_title('Regresión \n Salinidad vs Temperatura \n (por temporada)')
g1.ax.grid(False)
out1 = os.path.join(path_imagenes, 'lm_salinidad_vs_temp_por_temporada.png')
g1.fig.savefig(out1, dpi=300, bbox_inches='tight')
print(f'Guardado: {out1}')
plt.show()

# DIC (y) vs Salinidad (x) con líneas por temporada
sub_ds = df_reg.dropna(subset=['sal_psu', 'dic_micromol_kg'])
g2 = sns.lmplot(
    data=sub_ds,
    x='sal_psu', y='dic_micromol_kg',
    hue='season',
    height=5, aspect=1.3,
    ci=95,
    scatter_kws=dict(alpha=0.5, s=60, edgecolor='none')
)
g2.set_axis_labels('Salinidad (PSU)', 'DIC (µmol/kg)')
g2.ax.set_title('Regresión \n DIC vs Salinidad \n (por temporada)')
g2.ax.grid(False)
out2 = os.path.join(path_imagenes, 'lm_dic_vs_salinidad_por_temporada.png')
g2.fig.savefig(out2, dpi=300, bbox_inches='tight')
print(f'Guardado: {out2}')
plt.show()

'''
#correlación_de_spearman
'''

# Correlación de Pearson: Salinidad vs Temperatura y DIC vs Salinidad
try:
    from scipy.stats import shapiro  # Verifica disponibilidad de SciPy y Shapiro–Wilk
except Exception as e:
    raise ImportError("Se requiere scipy para ejecutar la prueba Shapiro–Wilk. Instale con: pip install scipy") from e

def pearson_correlation(df, x_col, y_col, label_x=None, label_y=None, group=None):
    """
    Calcula la correlación de Pearson entre x_col e y_col.
    - Si group es None: devuelve una fila global.
    - Si group es una columna: devuelve una fila por cada grupo.
    """
    label_x = label_x or x_col
    label_y = label_y or y_col

    def _clean(_df):
        _tmp = _df[[x_col, y_col]].copy()
        _tmp[x_col] = pd.to_numeric(_tmp[x_col], errors='coerce')
        _tmp[y_col] = pd.to_numeric(_tmp[y_col], errors='coerce')
        return _tmp.dropna(subset=[x_col, y_col])

    rows = []
    if group is None:
        sub = _clean(df)
        n = len(sub)
        r, p = (np.nan, np.nan)
        if n >= 2:
            r, p = pearsonr(sub[x_col], sub[y_col])
        rows.append({
            'nivel': 'GLOBAL',
            'grupo': 'ALL',
            'x': label_x,
            'y': label_y,
            'n': n,
            'r': r,
            'p_value': p
        })
    else:
        for g, gdf in df.groupby(group, dropna=False):
            sub = _clean(gdf)
            n = len(sub)
            r, p = (np.nan, np.nan)
            if n >= 2:
                r, p = pearsonr(sub[x_col], sub[y_col])
            rows.append({
                'nivel': f'POR_{group.upper()}',
                'grupo': str(g),
                'x': label_x,
                'y': label_y,
                'n': n,
                'r': r,
                'p_value': p
            })
    return pd.DataFrame(rows)

# Ejecutar correlaciones globales
corr_global = pd.concat([
    pearson_correlation(df, 'sal_psu', 'temp_c', 'Salinidad (PSU)', 'Temperatura (°C)'),
    pearson_correlation(df, 'sal_psu', 'dic_micromol_kg', 'Salinidad (PSU)', 'DIC (µmol/kg)')
], ignore_index=True)

# Ejecutar correlaciones por temporada
corr_por_temporada = pd.concat([
    pearson_correlation(df, 'sal_psu', 'temp_c', 'Salinidad (PSU)', 'Temperatura (°C)', group='season'),
    pearson_correlation(df, 'sal_psu', 'dic_micromol_kg', 'Salinidad (PSU)', 'DIC (µmol/kg)', group='season')
], ignore_index=True)

# Guardar resultados
pearson_global_csv = os.path.join(path_resultados, 'pearson_global.csv')
pearson_temporada_csv = os.path.join(path_resultados, 'pearson_por_temporada.csv')
corr_global.to_csv(pearson_global_csv, index=False, encoding='utf-8')
corr_por_temporada.to_csv(pearson_temporada_csv, index=False, encoding='utf-8')

print('-' * 60)
print(f'Correlaciones de Pearson (global) guardadas en: {pearson_global_csv}')
print(f'Correlaciones de Pearson (por temporada) guardadas en: {pearson_temporada_csv}')







