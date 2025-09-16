# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:46 2024

@author: jrbof
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from matplotlib.gridspec import GridSpec

from funciones.modulo_funciones  import (makecar,
                                         process_data,
                                         load_and_process_data,
                                         )

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

# names of calibrated sensor datasets
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
x_sensores = np.sort(x_sensores)

x_sensores = np.concatenate(([x1[0]], x_sensores[:]))

for tipo in ['Reg']:
    print('----------------------------------------------------------------')
    print(f"Processing data --> {tipo}")
    
    # Load data for all cases
    df_lab = [load_and_process_data(dirparolejsenr, name_lis_caso[i], tipo) for i in range(3)]
    
    overall = process_data(df_lab, tipo)

    # Plotting H/Hrms
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.6), dpi=300, facecolor='w', edgecolor='k')
    #fig.suptitle(f"{tipo} \n", fontsize=18, y=0.95)
    
    gs = GridSpec(3, 3, figure=fig)
    gs.update(wspace=0.5, hspace=3.5)
    
    for i, ax in enumerate(axes[0, :]):
        
        if tipo == 'Reg':
            ax.scatter(x_sensores[:], df_lab[i][:, 4], s=100, color='b', edgecolor='w', zorder=3, label='H')
            
        else:
            ax.scatter(x_sensores[:], df_lab[i][:, 12], s=100, color='b', edgecolor='w', zorder=3, label='Hrms')
        
        if i == 0:
            ax.set_ylabel('H [m]' if tipo == 'Reg' else 'Hrms [m]', fontsize=18)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        ax.text(0.10, 0.999, f"({chr(65 + i*2).lower()})", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right', fontstyle='italic', fontweight='bold')
        
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        ax.set_ylim(0 if tipo == 'Reg' else 0, 
                        overall[0,1] + 0.02 if tipo == 'Reg' else overall[1,1] + 0.02)
        
        if i == 0:  # Add legend only to the first subplot
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower left', ncol=2, fontsize=16)
        
        if i > 0:
            ax.set_yticks([])
        
        ax.set_xticks([])
        
    for i, ax in enumerate(axes[1, :]):
        ax.scatter(x_sensores[:], df_lab[i][:, 2] - df_lab[i][0, 2], s=100, color='navy', edgecolor='w', zorder=3, label=r'$\overline{\eta}$')
        
        ax.set_xlabel('x [m]', fontsize=18)
        
        if i == 0:
            ax.set_ylabel(r'$\overline{\eta}$ [m]', fontsize=18)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        ax.text(0.10, 0.999, f"({chr(66 + i*2).lower()})", transform=ax.transAxes, fontsize=20, verticalalignment='top', horizontalalignment='right', fontstyle='italic', fontweight='bold')
        
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        
        lim = -0.003 if tipo == 'Reg' else -0.0005
        ax.set_ylim(overall[2,0] + lim, overall[2,1] + 0.001)
        
        if i > 0:
            ax.set_yticks([])
    
    # Create a single legend for all subplots
    handles, labels = [], []
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll and ll not in labels:
                handles.append(hh)
                labels.append(ll)

    if handles:
        fig.subplots_adjust(bottom=0.18)
        fig.legend(handles, labels, loc='lower center', ncol=max(1, len(labels)), fontsize=20, frameon=True)

    plt.savefig(os.path.join(dirimg, f'Sensors_comparison_{tipo}.png'), bbox_inches='tight', pad_inches=0.25)
    plt.clf()
    plt.close()



