# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:56:51 2024

@author: jrbof
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from scipy.optimize import newton
from scipy import stats

# Disable seaborn style: revert to Matplotlib defaults
sns.reset_orig()
plt.style.use('default')

'''
---------------------------------------------------------------------------------------------------
    makecar --> create a folder in directory with case name

   Input:
    dir --> directory where the folder will be created
    name --> folder name

---------------------------------------------------------------------------------------------------
'''
def makecar(dir, name):
    # create the full path of the folder
    folder_path = os.path.join(dir, name)

    # try to create the folder if it doesn't exist
    try:
        os.makedirs(folder_path, exist_ok=True)
        #print(f"Folder created or already exists: {folder_path}")
    except Exception as e:
        print(f"Could not create folder: {folder_path}. Error: {e}")

    '''
---------------------------------------------------------------------------------------------------
    format_fixed_width --> format header and columns to a fixed width, handling NaN with blanks

   Inputs:
    df --> DataFrame to format
    width --> column width

   Output:
    formatted_df --> DataFrame with fixed-width formatting
---------------------------------------------------------------------------------------------------
    '''
def format_fixed_width(df, width=12):

    # replace NaN with a blank string matching the numeric width
    df = df.fillna(' ' * width)

    # format column names (headers)
    df.columns = [f"{col:<{width}}" for col in df.columns]  # left align

    # format each value in the DataFrame cells
    formatted_df = df.map(lambda x: f"{x:>{width}}" if isinstance(x, str) else f"{x:>{width}.10f}")

    return formatted_df

'''
---------------------------------------------------------------------------------------------------
    load_and_process_data --> load and process data from a CSV file

   Inputs:
    caso_path --> directory where the data is located
    name_lis_caso --> case list name
    name_caso --> file name without extension

   Output:
    df --> numpy array with loaded and processed data
---------------------------------------------------------------------------------------------------

'''
def load_and_process_data(caso_path, name_lis_caso, name_caso):
    file_path = os.path.join(caso_path,
                             name_lis_caso + '_' + name_caso + '.csv')
    df = pd.read_csv(file_path, header=0, sep='\t').replace(r'^\s*$', np.nan, regex=True).to_numpy()
    return np.array(df, dtype=float)


"""
---------------------------------------------------------------------------------------------------
Find the best t0 and the number of complete periods N.

   Inputs:
    T (float): Period [s]
    t_end (float): Final time [s]
    t_target (float): Approximate desired start time [s]

   Output:
    float: Best t0
---------------------------------------------------------------------------------------------------
"""
def find_best_t0(T, t_target, tipo, t_end=480):
    # possible values of N (number of integer periods up to t_end)
    N_vals = np.arange(0, int(np.floor(t_end / T)) + 1)

    # for each N, compute possible t0
    t0_candidates = t_end - N_vals * T

    # keep only t0 >= 0
    valid = t0_candidates >= 0
    N_vals = N_vals[valid]
    t0_candidates = t0_candidates[valid]

    # choose the candidate closest to t_target
    idx_best = np.argmin(np.abs(t0_candidates - t_target))
    N_best = N_vals[idx_best]
    t0_best = t0_candidates[idx_best]
    difference = abs(t0_best - t_target)

    #print(f'{tipo}')
    print(f"Best N = {N_best} complete periods")
    print(f"→ t_start = {t0_best:.2f} s (difference {difference:.3f} s)")

    return t0_best

"""
---------------------------------------------------------------------------------------------------
    Calculate the number of complete cycles and the total time.

   Inputs:
    T (float): Period in seconds.

   Output:
    float: Total time (t_end).
---------------------------------------------------------------------------------------------------
"""
def calculate_period_and_time(T, tipo):

    N = int(np.floor(20 / T))  # complete cycles
    t_end = N * T              # seconds

    #print('+'*64)
    print(f'{tipo}')
    print(f"Best N = {N} complete periods")
    print(f"→ rest_time = {t_end:.2f} s")

    return t_end

'''
---------------------------------------------------------------------------------------------------
    process_data --> process data and compute global min/max ignoring NaNs

   Inputs:
    casos --> list of numpy matrices with data from different cases
    tipo --> wave type (regular or irregular)

   Output:
    matrix with global min/max for H, Hrms, corrected eta and Sxx
---------------------------------------------------------------------------------------------------

'''
def process_data(casos, tipo):
    # Compute global min/max across all cases, ignoring NaNs

    # H (col 4)
    min_H = np.nanmin([np.nanmin(df[:, 4]) for df in casos])
    max_H = np.nanmax([np.nanmax(df[:, 4]) for df in casos])

    # Hrms (col 12)
    min_Hrms = np.nanmin([np.nanmin(df[:, 12]) for df in casos])
    max_Hrms = np.nanmax([np.nanmax(df[:, 12]) for df in casos])

    # eta corrected (col 2, subtract first value per case)
    min_eta = np.nanmin([np.nanmin(df[:, 2] - df[0, 2]) for df in casos])
    max_eta = np.nanmax([np.nanmax(df[:, 2] - df[0, 2]) for df in casos])

    # Sxx (col 17)
    min_Sxx = np.nanmin([np.nanmin(df[:, 17]) for df in casos])
    max_Sxx = np.nanmax([np.nanmax(df[:, 17]) for df in casos])

    return np.array([
        [min_H,   max_H],
        [min_Hrms, max_Hrms],
        [min_eta, max_eta],
        [min_Sxx, max_Sxx]
    ], dtype=float)

'''
def calcular_hrms(valores):
    """
    Computes H_rms for a list of values using:
    H_rms = sqrt((1/N) * sum(H_i^2))
    """
    N = len(valores)
    if N == 0:
        raise ValueError("The list of values is empty")
    
    suma_cuadrados = sum(h**2 for h in valores)
    hrms = np.sqrt(suma_cuadrados / N)
    return hrms
'''
def calcular_hrms(valores):
    # Root-mean-square of an array of values
    arr = np.array(valores)
    return np.sqrt(np.mean(arr ** 2))


'''
---------------------------------------------------------------------------------------------------
    calcular_parametros_oleaje_caso_agrupado: compute wave parameters for grouped study case

   Input:
    data --> sensor data
    x1 --> x positions
    name_caso --> case name
    dirimg --> directory to save images
    dirparolejmod --> directory to save wave parameters
    tipo --> wave type (regular or irregular)
    time_pause_reposo --> rest time in seconds
    time_inicio --> start time of data to analyze, seconds
    time_pause --> end time of data to analyze, seconds

   Output:
    csv file with wave parameters
    wave parameters plot
---------------------------------------------------------------------------------------------------
'''
def calcular_parametros_oleaje_caso_agrupado(data, x1, name_caso, dirimg, dirparolejmod, tipo, time_pause_reposo, time_inicio, time_pause, rho=999, g=9.81):

    print(f"{'-'*64}\nProcessing data --> {name_caso} --> {tipo}")

    no_wave = []   # number of waves
    Hmed = []      # mean height
    Tmed = []      # mean period
    Hs = []        # significant height
    T_Hs = []      # significant period
    Ts = []        # significant period associated to Hs values
    Hrms = []      # root-mean-square wave height
    Hrms_dvs = []  # significant height of the standard deviation
    Hmax = []      # maximum height
    T_Hmax = []    # period associated to maximum height

    sl = []  # free surface
    mcwl = []  # water level

    # Apply Savitzky-Golay filter to smooth the data in df_lab
    window_length = 20  # Choose an odd number for the window length
    polyorder = 3       # Polynomial order for the filter

    from scipy.signal import savgol_filter
    data[:, 1:] = savgol_filter(data[:, 1:],
                                window_length=window_length,
                                polyorder=polyorder, axis=0)

    # select time data
    tiempo = data[:, 0]

    # select data up to a given rest time
    df_lab_pause_reposo = tiempo[tiempo[:] <= time_pause_reposo]
    pause_reposo = df_lab_pause_reposo.shape[0]

    # select data for start of calculation
    df_lab_inicio = tiempo[tiempo[:] <= time_inicio]
    inicio = df_lab_inicio.shape[0]

    # select data up to a given time
    df_lab_pause = tiempo[tiempo[:] <= time_pause]
    pause = df_lab_pause.shape[0]

    print(f"Rest pause samples: {pause_reposo} -- > {time_pause_reposo}")
    print(f"Start samples: {inicio} -- > {time_inicio}")
    print(f"End samples: {pause} --> {time_pause}")

    for i in range(1, data.shape[1] - 1):
        nivel_reposo = np.mean(data[:pause_reposo, i])
        eta = np.mean(data[inicio:pause, i]) - nivel_reposo

        diff = (data[inicio:pause, i] - nivel_reposo) - eta
        cdxc = zerodown_gmm(diff, tiempo[inicio:pause])

        # Sort cdxc[0][:, 1] descending
        sorted_indices = np.argsort(cdxc[0][:, 1])[::-1]
        sorted_heights = cdxc[0][sorted_indices, 1]
        sorted_periods = cdxc[0][sorted_indices, 2]

        # Mean of the upper third of sorted_heights
        H_s = np.mean(sorted_heights[:max(1, len(sorted_heights) // 3)])
        T_H_s = np.mean(sorted_periods[:max(1, len(sorted_periods) // 3)])  # significant period
        T_s = np.mean(np.sort(cdxc[0][:, 2])[::-1][:max(1, cdxc[1] // 3)])  # significant period associated to Hs

        # Hrms and Hrms_dvs
        H_rms_dvs = 4 * np.std(diff)
        H_rms = calcular_hrms(cdxc[0][:, 1])

        # mean values of H and T
        H_med = np.mean(sorted_heights)
        T_med = np.mean(sorted_periods)

        # maximum H
        H_max = sorted_heights[0]
        T_max = sorted_periods[0]

        '''
        # assign values
        '''
        no_wave.append(cdxc[1])

        sl.append(eta)
        mcwl.append(nivel_reposo)

        Hmed.append(H_med)
        Tmed.append(T_med)

        Hs.append(H_s)
        T_Hs.append(T_H_s)  # significant period
        Ts.append(T_s)      # significant period associated to Hs

        Hrms.append(H_rms)
        Hrms_dvs.append(H_rms_dvs)

        Hmax.append(H_max)
        T_Hmax.append(T_max)

    '''
    # arrays to numpy
    '''
    no_wave = np.array(no_wave)

    sl = np.array(sl)
    mcwl = np.array(mcwl)

    Hmed = np.array(Hmed)
    Tmed = np.array(Tmed)

    Hs = np.array(Hs)
    T_Hs = np.array(T_Hs)
    Ts = np.array(Ts)

    Hrms = np.array(Hrms)
    Hrms_dvs = np.array(Hrms_dvs)

    Hmax = np.array(Hmax)
    T_Hmax = np.array(T_Hmax)

    # h0 and d
    h0 = h0_lab(x1)

    d_values = h0 + sl

    '''
    # wavelength
    '''
    Lmed = np.array(parallel_dispersion(Tmed, h0, num_cores=6))    # mean wavelength, associated to Tmed
    L_Hs = np.array(parallel_dispersion(T_Hs, h0, num_cores=6))    # significant wavelength
    Ls = np.array(parallel_dispersion(Ts, h0, num_cores=6))        # significant wavelength, associated to Hs
    Lmax = np.array(parallel_dispersion(T_Hmax, h0, num_cores=6))  # maximum wavelength, associated to T_Hmax

    '''
    # Sxx TL 2nd order
    '''
    # Sxx_s = calcular_tensor_radiacion_hrms(h0, Hs, Ls)

    H, L = (Hmed, Lmed) if tipo.lower() == 'reg' else (Hrms, L_Hs)
    #print(tipo)
    Sxx_2do = calcular_tensor_radiacion_TL2doO(h0, H, L)

    # theoretical eta
    Sxx_ad = 0.1875 * rho * g * Hs**2
    Sxx_sp = 0.22 * rho * g * Hs**2

    # initialize arrays for eta values
    deta_dx = np.zeros(len(sl))
    deta_18 = np.zeros(len(sl))
    deta_22 = np.zeros(len(sl))

    deta_dx[0] = sl[0]
    deta_18[0] = sl[0]
    deta_22[0] = sl[0]

    for i in range(1, len(sl)):
        dSxx_dx = Sxx_2do[i] - Sxx_2do[i - 1]
        dSxx_dx_18 = Sxx_ad[i] - Sxx_ad[i - 1]
        dSxx_dx_22 = Sxx_sp[i] - Sxx_sp[i - 1]

        deta_dx[i] = deta_dx[i - 1] - (dSxx_dx / (rho * g * (h0[i - 1] + deta_dx[i - 1])))
        deta_18[i] = deta_18[i - 1] - (dSxx_dx_18 / (rho * g * (h0[i - 1] + deta_18[i - 1])))
        deta_22[i] = deta_22[i - 1] - (dSxx_dx_22 / (rho * g * (h0[i - 1] + deta_22[i - 1])))

    # Sxx TL 2nd order computed iteratively from eta
    Sxx_tl2 = np.zeros_like(sl)
    Sxx_tl2[0] = Sxx_2do[0]

    for j in range(1, len(sl)):
        # centered derivative of eta
        deta_dx_tl2 = sl[j] - sl[j-1]

        Sxx_tl2[j] = Sxx_tl2[j-1] - deta_dx_tl2 * rho * g * (h0[j-1] + sl[j-1])

    '''
    Plot
    '''
    # scatter plots
    fig = plt.figure(figsize=(25, 15), dpi=300, facecolor='w', edgecolor='k')

    gs = GridSpec(6, 4, figure=fig)
    gs.update(wspace=0.3, hspace=1)

    fig.suptitle(name_caso + '\n' + tipo, fontsize=30, y=0.92)

    ax = fig.add_subplot(gs[:3, :])
    ax.scatter(x1[:], Hmed[:] if str(tipo).lower() == 'reg' else Hrms[:],
               s=300, color='b', edgecolor='w', zorder=3, label='H' if tipo.lower() == 'reg' else r'$H_{rms}$')

    ax.set_xlabel('x [m]', fontsize=28)
    ax.set_ylabel('H [m]' if tipo.lower() == 'reg' else r'$H_{rms}$ (m)', fontsize=28)

    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)

    ax1 = fig.add_subplot(gs[3:6, :])
    ax1.scatter(x1[:], sl[:] - sl[0], s=300, color='navy', edgecolor='w', zorder=3, label=r'$\overline{\eta}$')

    ax1.set_xlabel('x [m]', fontsize=28)
    ax1.set_ylabel(r'$\overline{\eta}$ [m]', fontsize=28)

    ax1.tick_params(axis='x', labelsize=26)
    ax1.tick_params(axis='y', labelsize=26)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.spines['left'].set_position(('outward', 10))
    ax1.spines['bottom'].set_position(('outward', 10))

    ax1.spines['bottom'].set_linewidth(3)
    ax1.spines['left'].set_linewidth(3)

    # unified legend for both subplots
    handles_top, labels_top = ax.get_legend_handles_labels()
    handles_bot, labels_bot = ax1.get_legend_handles_labels()
    fig.legend(
        handles_top + handles_bot,
        labels_top + labels_bot,
        loc='lower center',
        ncol=2,
        facecolor='w',
        title='Legend',
        title_fontsize=30,
        fontsize=36,
        bbox_to_anchor=(0.5, -0.1)
    )

    plt.savefig(os.path.join(dirimg, name_caso, name_caso + '_' + tipo + '.png'), bbox_inches='tight', pad_inches=0.25)
    plt.clf()
    plt.close()

    data = {
        "mcwl [m]": np.nan_to_num(mcwl, nan=np.nan),
        "eta_mean [m]": np.nan_to_num(sl, nan=np.nan),
        "No.waves": np.nan_to_num(no_wave, nan=np.nan).astype(int),
        "Hmean [m]": np.nan_to_num(Hmed, nan=np.nan),
        "Tmean [s]": np.nan_to_num(Tmed, nan=np.nan),
        "Lmean [m]": np.nan_to_num(Lmed, nan=np.nan),
        "Hs [m]": np.nan_to_num(Hs, nan=np.nan),
        "T_s [s]": np.nan_to_num(T_Hs, nan=np.nan),
        "L_s [m]": np.nan_to_num(L_Hs, nan=np.nan),
        "T_Hs [s]": np.nan_to_num(Ts, nan=np.nan),
        "L_Hs [m]": np.nan_to_num(Ls, nan=np.nan),
        "Hrms [m]": np.nan_to_num(Hrms, nan=np.nan),
        "Hrms_std [m]": np.nan_to_num(Hrms_dvs, nan=np.nan),
        "Hmax [m]": np.nan_to_num(Hmax, nan=np.nan),
        "T_Hmax [s]": np.nan_to_num(T_Hmax, nan=np.nan),
        "Lmax [m]": np.nan_to_num(Lmax, nan=np.nan),
        "Sxx_TL2nd [J]": np.nan_to_num(Sxx_2do, nan=np.nan),
        "Sxx_TL_eta [J]": np.nan_to_num(Sxx_tl2, nan=np.nan),
        "Sxx_P18 [J]": np.nan_to_num(Sxx_ad, nan=np.nan),
        "Sxx_P22 [J]": np.nan_to_num(Sxx_sp, nan=np.nan),
        "eta_TL2nd [m]": np.nan_to_num(deta_dx, nan=np.nan),
        "eta_P18 [m]": np.nan_to_num(deta_18, nan=np.nan),
        "eta_P22 [m]": np.nan_to_num(deta_22, nan=np.nan),
        "h0 [m]": np.nan_to_num(h0, nan=np.nan),
        "d [m]": np.nan_to_num(d_values, nan=np.nan),
        "Hmean/h0": np.nan_to_num(Hmed / h0, nan=np.nan),
        "Hs/h0": np.nan_to_num(Hs / h0, nan=np.nan),
        "Hrms/h0": np.nan_to_num(Hrms / h0, nan=np.nan)
    }

    df = pd.DataFrame(data)
    df.index = [f"{i + 5:02d}" for i in range(1, len(df) + 1)]
    formatted_df = format_fixed_width(df)

    tipo = 'Reg' if str(tipo).lower() == 'reg' else 'Irrg'

    formatted_df.to_csv(os.path.join(dirparolejmod, name_caso + '_' + tipo + '.csv'), index=True, header=True, sep='\t')

"""
---------------------------------------------------------------------------------------------------
   zerodown_gmm: compute zero-down crossing wave height from data

   Input:
    t --> time vector
    f --> single column data

   Output:
    height --> time series of wave height
    amp_c --> time series of crest wave amplitude
    amp_t --> time series of trough wave amplitude
    period --> time series of wave period
    nw --> number of waves

   Transforming Nobuhito Mori code from .mat to .py
   Updated by Boffill 2025/02/14 output as an array
---------------------------------------------------------------------------------------------------
"""

def zerodown_gmm(f, t):
    """
    Version 1.2
    Optimized version to detect zero-down crossings and compute wave parameters.

    Parameters:
        f (np.array): Free surface elevation signal (1D).
        t (np.array): Associated time vector (1D).

    Returns:
        tuple: (parameter array, number of waves)
               array columns: [indices, height, period, amp_c, amp_t]
    """
    # Detect zero-down crossings (positive to negative)
    crossing_mask = (f[:-1] > 0) & (f[1:] <= 0)
    crossing_indices = np.where(crossing_mask)[0]

    # If there are not enough crossings, return NaN
    if len(crossing_indices) < 2:
        return np.full((1, 5), np.nan), 0

    # Compute exact crossing times by linear interpolation
    t0 = t[crossing_indices] - (t[crossing_indices+1] - t[crossing_indices]) * \
         f[crossing_indices] / (f[crossing_indices+1] - f[crossing_indices])

    # Segments between consecutive crossings
    start_indices = crossing_indices[:-1]
    end_indices = crossing_indices[1:]

    # Compute parameters per segment
    heights = []
    crests = []
    troughs = []
    periods = []

    for i in range(len(start_indices)):
        seg_start = start_indices[i] + 1
        seg_end = end_indices[i] + 1

        segment = f[seg_start:seg_end]
        crest = np.max(segment)
        trough = np.min(segment)

        heights.append(crest - trough)
        crests.append(crest)
        troughs.append(trough)
        periods.append(t0[i+1] - t0[i])

    # Build output array
    if not heights:
        return np.full((1, 5), np.nan), 0

    output = np.column_stack((
        crossing_indices[1:],  # crossing indices
        heights,
        periods,
        crests,
        troughs
    ))

    return output, len(heights)

'''
***************************************************************************************************
'''
def calculate_rms(values):
    '''
    calculate_rms --> compute Hrms

   Inputs:
    values --> array of wave heights

   Output:
    rms value
    '''
    # convert to numpy array for efficiency
    values = np.array(values)

    # compute RMS
    rms = np.sqrt(np.mean(values**2))

    return rms

'''
------------------------------------------------------------------------------------------------
Compute wavelength (L) from period (T) and depth (h) using the dispersion equation.

   Input:
        T: Period in seconds
        h: Depth in meters

   Output:
        L: Wavelength in meters
------------------------------------------------------------------------------------------------
'''
def dispersion(T, h):
    g = 9.81
    L0 = (g * T**2) / (2 * np.pi)
    equation = lambda L: L - L0 * np.tanh(2 * np.pi * h / L)
    return newton(equation, L0, tol=1e-6)

'''
------------------------------------------------------------------------------------------------
Parallelize wavelength calculation for multiple (T, h) pairs.

   Input:
        T_values: Sequence of periods
        h_values: Sequence of depths
        num_cores: Number of CPU cores to use (optional)

   Output:
        List of computed wavelengths
------------------------------------------------------------------------------------------------
'''

def parallel_dispersion(T_values, h_values, num_cores=None):
    T_h_pairs = list(zip(T_values, h_values))

    if num_cores is None:
        num_cores = cpu_count()  # use all available cores if not specified

    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(delayed(dispersion)(T, h) for T, h in T_h_pairs)

    return results


'''
------------------------------------------------------------------------------------------------
Compute radiation stress Sxx from period (T), depth (h) and wave height (Hs).

   Input:
    L: Wavelength
    h: Depth
    H: Wave height

   Output:
    Sxx: Radiation stress
------------------------------------------------------------------------------------------------
'''
def calcular_tensor_radiacion(h, H, L):
    #L = dispersion(T, h)

    def G(L, h):
        k = (2 * np.pi) / L
        return (2 * k * h) / (np.sinh(2 * k * h))

    def E(H):
        ro = 999
        grv = 9.81
        return (1 / 16) * (ro * grv * pow(H, 2))

    G_val = G(L, h)
    E_val = E(H)
    Sxx = (E_val / 2) * (1 + 2 * G_val)

    return Sxx

"""
------------------------------------------------------------------------------------------------
Compute radiation stress Sxx from wavelength (L), depth (h) and Hrms.

   Input:
    L: Wavelength
    h: Still water depth (h0)
    H: Root-mean-square wave height (Hrms)

   Output:
    Sxx: Radiation stress
------------------------------------------------------------------------------------------------
"""
def calcular_tensor_radiacion_hrms(h, H, L):

    #L = dispersion(T, h)

    def G(L, h):
        k = (2 * np.pi) / L
        return (2 * k * h) / (np.sinh(2 * k * h))

    def E(H):
        ro = 999
        grv = 9.81
        return (1 / 16) * (ro * grv * pow(H, 2))

    G_val = G(L, h)
    E_val = E(H)
    Sxx_hrms = (E_val) * (1/2 + G_val)

    return Sxx_hrms

"""
------------------------------------------------------------------------------------------------
Compute radiation stress (Sxx) based on 2nd-order Linear Theory (TL2nd), from wavelength (L),
depth (h) and wave height (H).

   Input:
    L: Wavelength
    h: Still water depth (h0)
    H: Wave height (H) (Hs for irregular, H for regular)
    tipo: Calculation type (irregular or regular) in case of Hs-based computation

   Output:
    Sxx: Radiation stress TL2nd
------------------------------------------------------------------------------------------------
    """
def calcular_tensor_radiacion_TL2doO(h, H, L):
    #L = dispersion(T, h)

    def G(L, h):
        """
        G = (2*k*h) / sinh(2*k*h).
        """
        k = (2 * np.pi) / L
        argumento = 2 * k * h
        resultado = argumento / np.sinh(argumento)
        return resultado

    #def E(H, tipo):
    def E(H):
        rho = 999
        g = 9.81
        #factor = 1/8 if tipo.lower() == 'reg' else 1/16
        factor = 1/8
        return factor * (rho * g * pow(H, 2))

    G_val = G(L, h)
    E_val = E(H)
    Sxx = (E_val) * (1/2 + G_val)

    return Sxx

'''
------------------------------------------------------------------------------------------------
h0_lab --> from sensor positions compute depth at sensor area

   Inputs:
    x_rest --> sensor positions

   Output:
    selected_points --> depth values h
------------------------------------------------------------------------------------------------
'''
def h0_lab(x_rest):
    # Generate bathymetry profile (triangle)
    triangle_x = np.linspace(0, 10, 100)
    triangle_y = np.linspace(-0.58, 0.42, 100)

    # Add an extra point to triangle_x and triangle_y
    triangle_x = np.append(0, triangle_x)
    triangle_y = np.append(0, triangle_y)

    # Interpolation of h points
    x = triangle_x
    y = triangle_y

    f_interpolated = interp1d(x, y, kind='linear')

    # generate new points for a smoother interpolation
    x_new = np.linspace(x.min(), x.max(), num=10000)
    y_new = f_interpolated(x_new)

    # keep points below zero
    mask = y_new < 0
    x_new = x_new[mask]
    y_new = y_new[mask]

    # select x_new points closest to x_rest, with the same number of values as x_rest
    selected_points = np.full((len(x_rest), 2), np.nan)  # initialize with NaNs to preserve length

    for i, xi in enumerate(x_rest):
        idx = (np.abs(x_new - xi)).argmin()  # closest index in x_new
        selected_points[i] = (x_new[idx], y_new[idx])

    # filter valid points (non-NaN)
    selected_points = selected_points[~np.isnan(selected_points).any(axis=1)]

    return selected_points[:,1]*(-1)


'''
---------------------------------------------------------------------------------------------------
_clean_vec --> clean vector by removing NaNs and infinities

   Input:
    a --> data vector

   Output:
    a --> cleaned vector
'''
def _clean_vec(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    return a[np.isfinite(a)]

'''
---------------------------------------------------------------------------------------------------
_normal_test --> perform normality tests (Shapiro-Wilk or Anderson-Darling)

   Input:
    x --> data vector

   Output:
    dictionary with normality test results
---------------------------------------------------------------------------------------------------
'''
def _normal_test(x):
    x = _clean_vec(x)
    n = x.size
    out = {"n": int(n), "method": None, "stat": np.nan, "p": np.nan, "ok": False}

    if n == 0:
        return out
    try:
        if n <= 2000:
            stat, p = stats.shapiro(x)
            out.update(method="Shapiro–Wilk", stat=float(stat), p=float(p), ok=bool(p > 0.05))
        else:
            ad = stats.anderson(x, dist="norm")
            levels = np.array(ad.significance_level)
            idx = int(np.argmin(np.abs(levels - 5.0)))
            crit = float(ad.critical_values[idx])
            ok = float(ad.statistic) < crit
            out.update(method="Anderson–Darling", stat=float(ad.statistic), p=np.nan, ok=bool(ok))
    except Exception:
        pass
    return out

'''---------------------------------------------------------------------------------------------------
_concat_for_mc --> concatenate data and labels for variance analysis

   Input:
    gs --> list of data vectors
    labels --> list of labels corresponding to each data vector

   Output:
    x --> concatenated data vector
    g --> concatenated label vector
---------------------------------------------------------------------------------------------------
'''
def _concat_for_mc(gs, labels):
    x = np.concatenate(gs)
    g = np.concatenate([np.repeat(labels[i], gs[i].size) for i in range(len(gs))])
    return x, g

'''
---------------------------------------------------------------------------------------------------
pick_target --> select target column in a DataFrame
    Input:
     df --> pandas DataFrame
     candidates --> list of candidate column names

    Output:
     selected column name or None if none found
---------------------------------------------------------------------------------------------------

'''
def pick_target(df, candidates):
        cols = df.columns
        for c in candidates:
            if c in cols:
                return c
        # flexible match
        for c in candidates:
            for col in cols:
                if str(c).lower() in str(col).lower():
                    return col
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return num_cols[-1] if num_cols else None


if __name__ == "__main__":
    pass
