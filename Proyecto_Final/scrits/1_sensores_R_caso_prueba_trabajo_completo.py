# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:09:46 2024

@author: jrbof
"""

import numpy as np
import os

from funciones.modulo_funciones  import (makecar,
                                         find_best_t0,
                                         calculate_period_and_time,
                                         calcular_parametros_oleaje_caso_agrupado)

# working directory
parentDirectory = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))

# calibrated sensor data name
config = {
    'lis_caso': ['H10T12', 'H10T25', 'H10T30'],
    'caso': ['Reg']
}

name_lis_caso = {i: name for i, name in enumerate(config['lis_caso'])}
name_caso = {i: oleaje for i, oleaje in enumerate(config['caso'])}

# sensor x-positions
x1 = np.array([0, 1.42, 0.93, 0.84, 0.20, 0.19, 0.20, 0.195, 0.20, 0.20, 0.23, 0.20, 0.20, 0.21])

x1[2:] = np.cumsum(x1[1:-1]) + x1[2:]
x2 = x1[1:] - 0.05
x3 = x1[1:] - 0.10

x_sensores = np.hstack((x1[1:], x2, x3))
x_sensores = np.sort(x_sensores)

x_sensores = np.concatenate(([x1[0]], x_sensores[:]))

# create directories
parentDirectory = os.path.join(parentDirectory)

makecar(parentDirectory, "images")
# directory to save images
dirimg = os.path.join(parentDirectory, "images")
# SensorsR directory
dirsenr = os.path.join(parentDirectory, "data")
# data processing directory
makecar(parentDirectory, "ParamtOleajSensoresR")
dirparolejsenr = os.path.join(parentDirectory, "ParamtOleajSensoresR")

# -------------------------------
# calculate parameters for cases grouped into a single case
# -------------------------------    
for le in range(3): #T12 / T25 / T30  

    print(f"{'~'*64} \n{'-'*64} \n {name_lis_caso[le]} \n{'-'*64}")
            
    makecar(dirimg, name_lis_caso[le])

    file_path = os.path.join(dirsenr, name_lis_caso[le], f'{name_lis_caso[le]}_processed_data.csv')
    data_caso = np.loadtxt(file_path, delimiter=',')
    # rest time
    t_reposo = round(calculate_period_and_time(int(name_lis_caso[le][-2:])/10, 'Reg'), 2)
    # start index for data processing
    inicio = 200 
    t0_best = round(find_best_t0(int(name_lis_caso[le][-2:])/10, inicio, 'Reg'), 2)
 
    calcular_parametros_oleaje_caso_agrupado(data_caso, 
                                            x_sensores, 
                                            name_lis_caso[le], 
                                            dirimg, 
                                            dirparolejsenr, 
                                            'Reg',
                                            t_reposo,
                                            t0_best,
                                            480)
print(f"{'~'*64} \n")
# -------------------------------