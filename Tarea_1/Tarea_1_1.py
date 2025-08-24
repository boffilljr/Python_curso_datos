"""
@author: JOSE ROLANDO BOFFILL VAZQUEZ
"""


'''
 Imprimir "¡Hola, mundo!"
 '''
 
print("¡Hola, mundo! / ¡Hello world!")



'''
Operaciones aritméticas simples 
'''

def calculadora():
    suma = lambda a, b: a + b
    resta = lambda a, b: a - b
    multiplicacion = lambda a, b: a * b
    division = lambda a, b: a / b

    ops = {
        '1': ('suma', suma),
        '2': ('resta', resta),
        '3': ('multiplicación', multiplicacion),
        '4': ('división', division),
        '+': ('suma', suma),
        '-': ('resta', resta),
        '*': ('multiplicación', multiplicacion),
        '/': ('división', division),
        'suma': ('suma', suma),
        'resta': ('resta', resta),
        'multiplicacion': ('multiplicación', multiplicacion),
        'multiplicación': ('multiplicación', multiplicacion),
        'division': ('división', division),
        'división': ('división', division),
    }
    try:
        op_input = input("Seleccione operación (1:+ suma, 2:- resta, 3:* multiplicación, 4:/ división): ").strip().lower()
        if op_input not in ops:
            raise KeyError
        nombre_op, func = ops[op_input]
        a = float(input("Ingrese el primer número: ").strip())
        b = float(input("Ingrese el segundo número: ").strip())
        if nombre_op == 'división' and b == 0:
            print("Error: división por cero.")
            return
        resultado = func(a, b)
        print(f"Operacion aritmética realizada: {nombre_op} -- > Resultado: {resultado}")
    except ValueError:
        print("Entrada inválida. Ingrese números válidos.")
    except KeyError:
        print("Operación inválida. Elija una opción válida.")

if __name__ == "__main__":
    calculadora()

