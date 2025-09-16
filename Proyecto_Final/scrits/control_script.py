# -*- coding: utf-8 -*-
"""
Script de control para ejecución de análisis de datos de sensores.
Permite seleccionar qué script ejecutar, asegurando el orden correcto.
"""

import os
import sys
import subprocess
import time
import codecs



# Colores para la salida en terminal (usando códigos ANSI simples)
class Colors:
    HEADER = ''
    BLUE = ''
    GREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''
    BOLD = ''
    UNDERLINE = ''

# Para Windows, intentar activar soporte para ANSI escape codes
if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        
        # Ahora podemos usar colores ANSI en Windows
        Colors.HEADER = '\033[95m'
        Colors.BLUE = '\033[94m'
        Colors.GREEN = '\033[92m'
        Colors.WARNING = '\033[93m'
        Colors.FAIL = '\033[91m'
        Colors.ENDC = '\033[0m'
        Colors.BOLD = '\033[1m'
        Colors.UNDERLINE = '\033[4m'
    except:
        # Si falla, no usamos colores
        pass
#aaaa
#aaaa
####################################
def clear_screen():
    """Clears the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Prints the menu header"""
    clear_screen()
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print(f"{'='*64}")
    print("          SENSOR DATA ANALYSIS SYSTEM          ")
    print(f"{'='*64}")
    print(f"{Colors.ENDC}")

def print_menu():
    """Prints the options menu"""
    print(f"{Colors.BOLD}Available options:{Colors.ENDC}")
    print(f"{Colors.BLUE}1{Colors.ENDC}. Sensors_R")
    print(f"{Colors.BLUE}2{Colors.ENDC}. Descriptive statistics")
    print(f"{Colors.BLUE}3{Colors.ENDC}. Plot data")
    print(f"{Colors.BLUE}4{Colors.ENDC}. Assumption check (normality and homoscedasticity)")
    print(f"{Colors.BLUE}5{Colors.ENDC}. Parametric test")
    print(f"{Colors.BLUE}6{Colors.ENDC}. Non-parametric test")
    print(f"{Colors.BLUE}7{Colors.ENDC}. Machine Learning")
    print(f"{Colors.BLUE}8{Colors.ENDC}. Run all options in order")
    print(f"{Colors.BLUE}0{Colors.ENDC}. Exit")
    print("")

def check_file_exists(filename):
    """Checks if a file exists"""
    return os.path.isfile(filename)

def run_script(script_name):
    """Executes a script and shows real-time output"""
    if not check_file_exists(script_name):
        print(f"{Colors.FAIL}Error: File {script_name} does not exist.{Colors.ENDC}")
        input("Press Enter to continue...")
        return False
    
    print(f"{Colors.GREEN}Running {script_name}...{Colors.ENDC}")
    print("-" * 50)
    
    try:
        # Configure environment to support UTF-8
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the script in text mode with UTF-8 encoding
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,  # Text mode
            encoding='utf-8',        # Explicit encoding
            errors='replace',        # Replace invalid characters
            bufsize=1,               # Line buffering (valid in text mode)
            env=env
        )
        
        # Read and show output in real time
        for line in process.stdout:
            # Replace problematic characters
            print(line.replace('\u2192', '->'), end='', flush=True)
        
        return_code = process.wait()
            
        if return_code == 0:
            print(f"{Colors.GREEN}Execution completed successfully.{Colors.ENDC}")
            return True
        else:
            print(f"{Colors.FAIL}Execution failed (code: {return_code}).{Colors.ENDC}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{Colors.FAIL}Error: Execution time exceeded for {script_name}.{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.FAIL}Unexpected error: {str(e)}{Colors.ENDC}")
        return False
    
    input("Press Enter to continue...")
    return True


def main():
    """Main function"""
    # Map options to scripts
    scripts = {
        '1': '1_sensores_R_caso_prueba_trabajo_completo.py',
        '2': '2_estadística_descriptiva.py',
        '3': '2_grafico_datos.py',
        '4': '3_supuestos_normalidad_homocedasticidad.py',
        '5': '4_prueba_parametrica.py',
        '6': '5_prueba_no_parametrica.py',
        '7': '6_ML.py'
    }
    
    # Verify that script 1 exists before continuing
    if not check_file_exists(scripts['1']):
        print(f"{Colors.FAIL}Error: The initial script {scripts['1']} does not exist.{Colors.ENDC}")
        print("This script is required to generate the initial data.")
        print("Make sure it is in the same directory as this control script.")
        sys.exit(1)
    
    # Flag to verify if data has been generated
    data_generated = False
    
    while True:
        print_header()
        print_menu()
        
        # If data has not been generated, show warning
        if not data_generated and any(check_file_exists(f) for f in ['data_raw.pkl', 'data_processed.pkl']):
            data_generated = True
            
        if not data_generated:
            print(f"{Colors.WARNING}WARNING: Data has not been generated yet.{Colors.ENDC}")
            print(f"{Colors.WARNING}You must run option 1 before using other options.{Colors.ENDC}")
            print("")
        
        opcion = input(f"{Colors.BOLD}Select an option (0-8): {Colors.ENDC}").strip()
        
        if opcion == '0':
            print("Exiting...")
            break
            
        elif opcion == '1':
            if run_script(scripts['1']):
                data_generated = True
                print("Data generated successfully.")
            input("Press Enter to continue...")
            
        elif opcion in ['2', '3', '4', '5', '6', '7']:
            if not data_generated:
                print(f"{Colors.WARNING}Data has not been generated yet.{Colors.ENDC}")
                answer = input("Do you want to run option 1 first? (y/n): ").strip().lower()
                if answer == 'y':
                    if run_script(scripts['1']):
                        data_generated = True
                        print("Data generated successfully.")
                        # Now run the selected option
                        run_script(scripts[opcion])
                else:
                    print("Continuing without generated data (errors may occur).")
                    run_script(scripts[opcion])
            else:
                run_script(scripts[opcion])
                
        elif opcion == '8':
            # Run all options in order
            print(f"{Colors.BOLD}Running all options in order...{Colors.ENDC}")
            
            # First run option 1
            if run_script(scripts['1']):
                data_generated = True
                print("Data generated successfully.")
                
                # Run the rest of the options
                for key in ['2', '3', '4', '5', '6', '7']:
                    print(f"\n{Colors.HEADER}Running {scripts[key]}...{Colors.ENDC}")
                    run_script(scripts[key])
                    time.sleep(1)  # Small pause between executions
                    
            input("Press Enter to continue...")
            
        else:
            print(f"{Colors.FAIL}Invalid option. Please select a number between 0 and 8.{Colors.ENDC}")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
