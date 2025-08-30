# -*- coding: utf-8 -*-
"""
Script con menú para ejecutar distintos módulos de análisis.

Opciones:
1) dataframes
2) estadísticas descriptivas de la gráfica_matplotib
3) estadísticas_descriptivas_seaborn
4) pruebas_de_normalidad
5) No paramétrico_Mann_whitney
6) No paramétrico_Kruskall-Wallis
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from scipy.stats import kruskal
from scipy.stats import shapiro

# -------------------------------
# Utilidades de paths y lectura
# -------------------------------
def setup_paths():
    wdir = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(wdir, 'data')

    path_imagenes = os.path.join(wdir, 'imagenes')
    os.makedirs(path_imagenes, exist_ok=True)

    path_resultados = os.path.join(wdir, 'resultados')
    os.makedirs(path_resultados, exist_ok=True)

    return wdir, path_data, path_imagenes, path_resultados


def load_data(path_data: str) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(path_data, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f'No se encontraron archivos .csv en {path_data}')
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')

    # Pre-cálculo útil para varios módulos
    df['TA_DIC_ratio'] = pd.to_numeric(df['ta_micromol_kg'], errors='coerce') \
        .div(pd.to_numeric(df['dic_micromol_kg'], errors='coerce'))
    df['TA_DIC_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# -------------------------------
# Módulo: funciones
# -------------------------------
def shapiro_test(df_in, col, nombre_variable, alpha=0.05, max_n=5000):
    
    serie = pd.to_numeric(df_in[col], errors='coerce').dropna()
    n_original = len(serie)
    if n_original < 3:
        print('-' * 60)
        print(f'Prueba Shapiro–Wilk: {nombre_variable}')
        print(f'No hay suficientes datos (n={n_original}) para ejecutar la prueba.')
        return {
            'variable': nombre_variable,
            'n_original': n_original,
            'n_analizado': n_original,
            'W': np.nan,
            'p_valor': np.nan,
            'decision(alpha=0.05)': 'insuficiente_n'
        }

    muestreo = False
    if n_original > max_n:
        serie = serie.sample(max_n, random_state=42)
        muestreo = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W, p_valor = shapiro(serie.to_numpy())

    decision = 'no_rechaza_normalidad' if p_valor >= alpha else 'rechaza_normalidad'

    print('-' * 60)
    print(f'Prueba Shapiro–Wilk: {nombre_variable}')
    print(f'n original: {n_original} | n analizado: {len(serie)}' + (' (muestreo a 5000)' if muestreo else ''))
    print(f'Estadístico W: {W:.4f} | p-valor: {p_valor:.4g}')
    print(f'Decisión (alpha={alpha}): {decision}')

    return {
        'variable': nombre_variable,
        'n_original': n_original,
        'n_analizado': len(serie),
        'W': W,
        'p_valor': p_valor,
        'decision(alpha=0.05)': decision
    }

def mannwhitney_by_season(df_in, col, nombre_variable, alpha=0.05):
    d = df_in.copy()
    d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna(subset=[col, 'season'])
    g_dry = d[d['season'].eq('Dry')][col].to_numpy()
    g_rainy = d[d['season'].eq('Rainy')][col].to_numpy()

    res = {
        'variable': nombre_variable,
        'n_Dry': int(g_dry.size),
        'n_Rainy': int(g_rainy.size),
        'U': np.nan,
        'p_valor': np.nan,
        'alternative': 'two-sided',
        'decision(alpha=0.05)': 'insuficiente_datos'
    }

    print('-' * 60)
    print(f'Prueba U de Mann-Whitney: {nombre_variable} (Dry vs Rainy)')
    print(f'n Dry: {g_dry.size} | n Rainy: {g_rainy.size}')

    if g_dry.size == 0 or g_rainy.size == 0:
        print('No hay suficientes datos en al menos uno de los grupos.')
        return res

    U, p = mannwhitneyu(g_dry, g_rainy, alternative='two-sided')
    res['U'] = float(U)
    res['p_valor'] = float(p)
    res['decision(alpha=0.05)'] = 'diferencia_significativa' if p < alpha else 'no_diferencia_significativa'

    print(f'U: {U:.4f} | p-valor: {p:.4g}')
    print(f'Decisión (alpha={alpha}): {res["decision(alpha=0.05)"]}')
    return res

def kruskal_by_group(df_in, col, group_col, nombre_variable, alpha=0.05):
    d = df_in.copy()
    d[col] = pd.to_numeric(d[col], errors='coerce')
    d = d.dropna(subset=[col, group_col])

    if d.empty:
        print('-' * 60)
        print(f'Kruskal–Wallis: {nombre_variable} por {group_col}')
        print('No hay datos válidos para la prueba.')
        return {
            'variable': nombre_variable,
            'factor': group_col,
            'k_grupos': 0,
            'gl': np.nan,
            'H': np.nan,
            'p_valor': np.nan,
            'decision(alpha=0.05)': 'sin_datos',
            'conteos_por_grupo': ''
        }

    counts = d.groupby(group_col)[col].size().sort_index()
    grupos_orden = counts.index.tolist()
    grupos = [d.loc[d[group_col] == g, col].to_numpy() for g in grupos_orden]

    print('-' * 60)
    print(f'Kruskal–Wallis: {nombre_variable} por {group_col}')
    print('Conteos por grupo:', dict(counts))

    if len(grupos) < 2:
        print('Se requieren al menos dos grupos para la prueba.')
        return {
            'variable': nombre_variable,
            'factor': group_col,
            'k_grupos': len(grupos),
            'gl': np.nan,
            'H': np.nan,
            'p_valor': np.nan,
            'decision(alpha=0.05)': 'grupos_insuficientes',
            'conteos_por_grupo': '; '.join([f'{g}:{int(n)}' for g, n in counts.items()])
        }

    H, p = kruskal(*grupos)
    decision = 'diferencias_significativas' if p < alpha else 'no_diferencias_significativas'
    print(f'H: {H:.4f} | gl: {len(grupos) - 1} | p-valor: {p:.4g}')
    print(f'Decisión (alpha={alpha}): {decision}')

    return {
        'variable': nombre_variable,
        'factor': group_col,
        'k_grupos': len(grupos),
        'gl': len(grupos) - 1,
        'H': float(H),
        'p_valor': float(p),
        'decision(alpha=0.05)': decision,
        'conteos_por_grupo': '; '.join([f'{g}:{int(n)}' for g, n in counts.items()])
    }

# -------------------------------
# Módulo: dataframes
# -------------------------------
def modulo_dataframes(df: pd.DataFrame, path_resultados: str):
    #dataframes por temporada
    df_dry = df[df['season'].eq('Dry')].copy()
    df_rainy = df[df['season'].eq('Rainy')].copy()

    #media y desviacion estandar de TA_DIC_ratio por temporada
    stats_TA_DIC_ratio = pd.DataFrame({
        'media': [df_dry['TA_DIC_ratio'].mean(skipna=True),
                  df_rainy['TA_DIC_ratio'].mean(skipna=True)],
        'desv_est': [df_dry['TA_DIC_ratio'].std(skipna=True),
                     df_rainy['TA_DIC_ratio'].std(skipna=True)]
    }, index=['DRY', 'RAINY'])

    print('-' * 60)
    print('TA_DIC_ratio - Media y desviación estándar por estación:')
    print('-' * 60)
    print(stats_TA_DIC_ratio)

    #media y desviacion estándar por season y area
    stats_TA_DIC_ratio_season_area = (
        df.groupby(['season', 'area'], dropna=False)['TA_DIC_ratio']
          .agg(media='mean', desv_est='std')
          .reset_index()
          .sort_values(['season', 'area'])
    )
    print('-' * 60)
    print('TA_DIC_ratio - Media y desviación estándar por season y area:')
    print('-' * 60)
    print(stats_TA_DIC_ratio_season_area)

    # Guardar Excel
    output_excel = os.path.join(path_resultados, 'TA_DIC_Season_Areas.xlsx')
    with pd.ExcelWriter(output_excel) as writer:
        stats_TA_DIC_ratio.to_excel(writer, sheet_name='stats_TA_DIC_ratio', index=True, index_label='season')
        stats_TA_DIC_ratio_season_area.to_excel(writer, sheet_name='stats_season_area', index=False)

    print('-' * 60)
    print(f'Resultados guardados en: {output_excel}')


# -------------------------------
# Módulo: estadísticas descriptivas de la gráfica_matplotib
# -------------------------------
def modulo_estadisticas_matplotlib(df: pd.DataFrame, path_imagenes: str):
    ta = pd.to_numeric(df['ta_micromol_kg'], errors='coerce')
    dic = pd.to_numeric(df['dic_micromol_kg'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(ta.dropna(), bins=30, ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('Histograma de TA')
    axes[0].set_xlabel('TA (µmol/kg)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(False)

    sns.histplot(dic.dropna(), bins=30, ax=axes[1], color='darkorange', edgecolor='white')
    axes[1].set_title('Histograma de DIC')
    axes[1].set_xlabel('DIC (µmol/kg)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(False)

    plt.tight_layout()
    fig_path = os.path.join(path_imagenes, 'hist_TA_DIC.png')
    plt.savefig(fig_path, dpi=300)
    print(f'Figura guardada en: {fig_path}')
    plt.show()

    #dispersion entre sal_psu, temp_c y depth
    df_scatter = df.copy()
    df_scatter['sal_psu'] = pd.to_numeric(df_scatter['sal_psu'], errors='coerce')
    df_scatter['temp_c'] = pd.to_numeric(df_scatter['temp_c'], errors='coerce')
    df_scatter['depth_m'] = pd.to_numeric(df_scatter['depth_m'], errors='coerce')
    df_scatter = df_scatter.dropna(subset=['sal_psu', 'temp_c', 'depth_m'])

    norm = mpl.colors.Normalize(vmin=df_scatter['depth_m'].min(), vmax=df_scatter['depth_m'].max())

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df_scatter,
        x='sal_psu',
        y='temp_c',
        hue='depth_m',
        hue_norm=norm,
        palette='viridis',
        style='season',
        alpha=0.5,
        s=100,
        edgecolor='none',
        ax=ax,
        legend='brief'
    )
    ax.grid(False)
    ax.set_title('Dispersión \n Salinidad (PSU) vs Temperatura (°C)')
    ax.set_xlabel('Salinidad \n (PSU)')
    ax.set_ylabel('Temperatura \n (°C)')

    sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15)
    cbar.set_label('Profundidad (m)')

    plt.tight_layout()
    scatter_path = os.path.join(path_imagenes, 'scatter_sal_psu_vs_temp_c_depth.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    print(f'Figura guardada en: {scatter_path}')
    plt.show()


# -------------------------------
# Módulo: estadísticas_descriptivas_seaborn
# -------------------------------
def modulo_estadisticas_seaborn(df: pd.DataFrame, path_imagenes: str):
    sns.set_theme(style='whitegrid')

    df_num = df.copy()
    df_num['ta_micromol_kg'] = pd.to_numeric(df_num['ta_micromol_kg'], errors='coerce')
    df_num['dic_micromol_kg'] = pd.to_numeric(df_num['dic_micromol_kg'], errors='coerce')

    fig_sns, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(
        data=df_num.dropna(subset=['ta_micromol_kg']),
        x='ta_micromol_kg', hue='season', bins=30, element='step',
        alpha=0.5, multiple='layer', ax=axes[0]
    )
    axes[0].set_title('TA por temporada \n (seaborn)')
    axes[0].set_xlabel('TA (µmol/kg)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(False)

    sns.histplot(
        data=df_num.dropna(subset=['dic_micromol_kg']),
        x='dic_micromol_kg', hue='season', bins=30, element='step',
        alpha=0.5, multiple='layer', ax=axes[1]
    )
    axes[1].set_title('DIC por temporada \n (seaborn)')
    axes[1].set_xlabel('DIC (µmol/kg)')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(False)

    plt.tight_layout()
    fig_sns_path = os.path.join(path_imagenes, 'sns_hist_TA_DIC_por_temporada.png')
    plt.savefig(fig_sns_path, dpi=300)
    print(f'Figura guardada en: {fig_sns_path}')
    plt.show()


# -------------------------------
# Módulo: pruebas_de_normalidad
'''
Se recomienda la prueba de Shapiro–Wilk para comprobar la normalidad en muestras pequeñas o moderadas, 
ya que es la más efectiva para detectar desviaciones de la normalidad. Por otro lado, la prueba de Kolmogorov–Smirnov 
es útil para determinar si una muestra proviene de una distribución continua específica, aunque no necesariamente normal.
'''
# -------------------------------
def modulo_pruebas_normalidad(df: pd.DataFrame, path_resultados: str):
    try:
        from scipy.stats import shapiro  # noqa: F401
    except Exception as e:
        print("Se requiere scipy para ejecutar la prueba Shapiro–Wilk. Instale con: pip install scipy")
        return

    resultados = []
    resultados.append(shapiro_test(df, 'dic_micromol_kg', 'DIC (µmol/kg)'))
    resultados.append(shapiro_test(df, 'sal_psu', 'Salinidad (PSU)'))

    res_df = pd.DataFrame(resultados)
    salida_csv = os.path.join(path_resultados, 'shapiro_normalidad_DIC_sal_psu.csv')
    res_df.to_csv(salida_csv, index=False, encoding='utf-8')
    print('-' * 60)
    print(f'Resultados Shapiro–Wilk guardados en: {salida_csv}')


# -------------------------------
# Módulo: No paramétrico_Mann_whitney
# -------------------------------
def modulo_mannwhitney(df: pd.DataFrame, path_imagenes: str, path_resultados: str):
    try:
        import scipy  # comprobacion rapida
    except Exception:
        print("Se requiere scipy para ejecutar Mann–Whitney. Instale con: pip install scipy")
        return

    resultados_mw = []
    resultados_mw.append(mannwhitney_by_season(df, 'dic_micromol_kg', 'DIC (µmol/kg)'))
    resultados_mw.append(mannwhitney_by_season(df, 'temp_c', 'Temperatura (°C)'))

    mw_df = pd.DataFrame(resultados_mw)
    mw_csv = os.path.join(path_resultados, 'mannwhitney_DIC_temp_por_temporada.csv')
    mw_df.to_csv(mw_csv, index=False, encoding='utf-8')
    print('-' * 60)
    print(f'Resultados Mann-Whitney guardados en: {mw_csv}')

    #boxplots y violin plots por temporada
    orden_season = ['Dry', 'Rainy']
    df_plot = df.copy()
    df_plot['dic_micromol_kg'] = pd.to_numeric(df_plot['dic_micromol_kg'], errors='coerce')
    df_plot['temp_c'] = pd.to_numeric(df_plot['temp_c'], errors='coerce')

    # Boxplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df_plot, x='season', y='dic_micromol_kg', order=orden_season, showfliers=False, ax=axes[0])
    axes[0].set_title('DIC por temporada (Boxplot)')
    axes[0].set_xlabel('Temporada')
    axes[0].set_ylabel('DIC (µmol/kg)')
    axes[0].grid(False)

    sns.boxplot(data=df_plot, x='season', y='temp_c', order=orden_season, showfliers=False, ax=axes[1])
    axes[1].set_title('Temperatura por temporada (Boxplot)')
    axes[1].set_xlabel('Temporada')
    axes[1].set_ylabel('Temperatura (°C)')
    axes[1].grid(False)

    plt.tight_layout()
    box_path = os.path.join(path_imagenes, 'boxplots_DIC_temp_por_temporada.png')
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    print(f'Figura guardada en: {box_path}')
    plt.show()

    #violin plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.violinplot(
        data=df_plot, x='season', y='dic_micromol_kg',
        order=orden_season, inner='quartile', cut=0, ax=axes[0]
    )
    axes[0].set_title('DIC por temporada (Violin)')
    axes[0].set_xlabel('Temporada')
    axes[0].set_ylabel('DIC (µmol/kg)')
    axes[0].grid(False)

    sns.violinplot(
        data=df_plot, x='season', y='temp_c',
        order=orden_season, inner='quartile', cut=0, ax=axes[1]
    )
    axes[1].set_title('Temperatura por temporada (Violin)')
    axes[1].set_xlabel('Temporada')
    axes[1].set_ylabel('Temperatura (°C)')
    axes[1].grid(False)

    plt.tight_layout()
    violin_path = os.path.join(path_imagenes, 'violin_DIC_temp_por_temporada.png')
    plt.savefig(violin_path, dpi=300, bbox_inches='tight')
    print(f'Figura guardada en: {violin_path}')
    plt.show()


# -------------------------------
# Módulo: No paramétrico_Kruskall-Wallis
# -------------------------------
def modulo_kruskal(df: pd.DataFrame, path_imagenes: str, path_resultados: str):
    try:
        import scipy  # comprobación rápida
    except Exception:
        print("Se requiere scipy para ejecutar Kruskal–Wallis. Instale con: pip install scipy")
        return

    #pruebas por area
    res_kw = []
    res_kw.append(kruskal_by_group(df, 'dic_micromol_kg', 'area', 'DIC (µmol/kg)'))
    res_kw.append(kruskal_by_group(df, 'temp_c', 'area', 'Temperatura (°C)'))

    kw_df = pd.DataFrame(res_kw)
    kw_csv = os.path.join(path_resultados, 'kruskal_area_DIC_temp.csv')
    kw_df.to_csv(kw_csv, index=False, encoding='utf-8')
    print('-' * 60)
    print(f'Resultados Kruskal–Wallis guardados en: {kw_df.shape} -> {kw_csv}')

    #boxplots por area
    df_area = df.copy()
    df_area['area'] = df_area['area'].astype(str)
    df_area['dic_micromol_kg'] = pd.to_numeric(df_area['dic_micromol_kg'], errors='coerce')
    df_area['temp_c'] = pd.to_numeric(df_area['temp_c'], errors='coerce')

    fig_kw, axes_kw = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=df_area.dropna(subset=['dic_micromol_kg', 'area']),
                x='area', y='dic_micromol_kg', showfliers=False, ax=axes_kw[0])
    axes_kw[0].set_title('DIC por área (Kruskal–Wallis)')
    axes_kw[0].set_xlabel('Área')
    axes_kw[0].set_ylabel('DIC (µmol/kg)')
    axes_kw[0].tick_params(axis='x', rotation=45)
    axes_kw[0].grid(False)

    sns.boxplot(data=df_area.dropna(subset=['temp_c', 'area']),
                x='area', y='temp_c', showfliers=False, ax=axes_kw[1])
    axes_kw[1].set_title('Temperatura por área (Kruskal–Wallis)')
    axes_kw[1].set_xlabel('Área')
    axes_kw[1].set_ylabel('Temperatura (°C)')
    axes_kw[1].tick_params(axis='x', rotation=45)
    axes_kw[1].grid(False)

    plt.tight_layout()
    box_kw_path = os.path.join(path_imagenes, 'boxplots_DIC_temp_por_area_KW.png')
    plt.savefig(box_kw_path, dpi=300, bbox_inches='tight')
    print(f'Figura guardada en: {box_kw_path}')
    plt.show()


# -------------------------------
# Menú
# -------------------------------
def mostrar_menu():
    print('\nSeleccione una opción:')
    print('1) dataframes')
    print('2) estadísticas descriptivas de la gráfica_matplotib')
    print('3) estadísticas_descriptivas_seaborn')
    print('4) pruebas_de_normalidad')
    print('5) No paramétrico_Mann_whitney')
    print('6) No paramétrico_Kruskall-Wallis')
    print('a) Ejecutar todas en orden (1->6)')
    print('q) Salir')


def main():
    wdir, path_data, path_imagenes, path_resultados = setup_paths()
    df = load_data(path_data)

    acciones = {
        '1': lambda: modulo_dataframes(df, path_resultados),
        '2': lambda: modulo_estadisticas_matplotlib(df, path_imagenes),
        '3': lambda: modulo_estadisticas_seaborn(df, path_imagenes),
        '4': lambda: modulo_pruebas_normalidad(df, path_resultados),
        '5': lambda: modulo_mannwhitney(df, path_imagenes, path_resultados),
        '6': lambda: modulo_kruskal(df, path_imagenes, path_resultados),
        'a': lambda: [acciones[str(i)]() for i in range(1, 7)],
    }

    while True:
        mostrar_menu()
        op = input('Opción: ').strip().lower()
        if op == 'q':
            print('Saliendo.')
            break
        accion = acciones.get(op)
        if not accion:
            print('Opción no válida.')
            continue
        accion()


if __name__ == '__main__':
    main()
