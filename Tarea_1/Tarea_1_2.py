"""
@author: JOSE ROLANDO BOFFILL VAZQUEZ
"""

   
'''
valor absoluto 
'''
def componer(f, g):
    return lambda x: f(g(x))

# |x| = sqrt(x*x) -> usando potencia 0.5
valor_absoluto_por_raiz = componer(lambda y: y ** 0.5, lambda x: x * x)

# |x| = max(x, -x)
valor_absoluto_por_max = componer(lambda t: max(t[0], t[1]),
                                      lambda x: (x, -x))

# |x| = x * sign(x), con sign(x) en {-1, 1}
_signo = lambda x: 1.0 if x >= 0 else -1.0
valor_absoluto_por_signo = componer(lambda pair: pair[0] * pair[1],
                                        lambda x: (x, _signo(x)))

# Solicitar número y mostrar resultados precedidos del método
try:
    x = float(input("Ingrese un número para calcular su valor absoluto: ").strip())
    metodos = [
            ("valor_absoluto_por_raiz", valor_absoluto_por_raiz),
            ("valor_absoluto_por_max", valor_absoluto_por_max),
            ("valor_absoluto_por_signo", valor_absoluto_por_signo),
        ]
    for nombre, f in metodos:
        print(f"{nombre} : {f(x)}")
except ValueError:
    print("Entrada inválida. Ingrese un número válido.")