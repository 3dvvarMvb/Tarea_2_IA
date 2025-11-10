"""
este modulo contiene funciones para generar graficos a partir de los resultados
de entrenamiento, incluyendo curvas de aprendizaje y matrices de confusion.
tambien incluye utilidades para detectar y verificar carpetas de ejecucion.
"""

import os, ast, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detectar_ultima():
    """
    detecta la ultima carpeta de ejecucion en el directorio 'resultados'.
    
    retorna:
    - ruta a la carpeta mas reciente que empiece con 'ejecucion-', o None si no hay.
    """
    base = "resultados"
    if not os.path.isdir(base):
        return None
    dirs = [d for d in os.listdir(base) if d.startswith("ejecucion-")]
    if not dirs:
        return None
    dirs.sort()
    return os.path.join(base, dirs[-1])

def normalizar_ruta(ruta: str):
    """
    normaliza una ruta de entrada, detectando automaticamente la ultima ejecucion si no se proporciona.
    
    parametros:
    - ruta: cadena con la ruta, o vacia para usar la ultima.
    
    retorna:
    - ruta absoluta normalizada.
    """
    ruta = (ruta or "").strip()
    if not ruta:
        return detectar_ultima()
    if os.path.isabs(ruta):
        return ruta
    # si no existe, probamos prefijar 'resultados/'
    if not os.path.exists(ruta):
        candidato = os.path.join("resultados", ruta)
        if os.path.exists(candidato):
            return candidato
    return ruta

def verificar_csvs(ruta):
    """
    verifica que existan los archivos csv necesarios en la ruta dada.
    
    parametros:
    - ruta: ruta a la carpeta de ejecucion.
    
    retorna:
    - lista de archivos faltantes.
    """
    faltan = []
    for nombre in ("registro_entrenamiento.csv", "resumen_top2.csv"):
        if not os.path.isfile(os.path.join(ruta, nombre)):
            faltan.append(nombre)
    return faltan

def curvas_entrenamiento(ruta_ejecucion: str):
    """
    genera y guarda un grafico con las curvas de exactitud de entrenamiento para cada candidato.
    
    parametros:
    - ruta_ejecucion: ruta a la carpeta con los resultados.
    
    guarda el grafico como 'curva_entrenamiento.png' y lo muestra.
    """
    log = pd.read_csv(os.path.join(ruta_ejecucion, "registro_entrenamiento.csv"))
    plt.figure()
    for nombre in sorted(log['nombre'].unique()):
        sub = log[log['nombre'] == nombre]
        plt.plot(sub['epoca'], sub['exactitud_entrenamiento'], label=nombre)
    plt.xlabel("epoca")
    plt.ylabel("exactitud (entrenamiento)")
    plt.title("evolucion de candidatos")
    plt.grid(True)
    plt.legend(loc="best")
    out = os.path.join(ruta_ejecucion, "curva_entrenamiento.png")
    plt.savefig(out, dpi=150)
    print("guardado:", out)
    plt.show()

def matrices_confusion(ruta_ejecucion: str):
    """
    genera y guarda graficos de matrices de confusion para los dos mejores candidatos.
    
    parametros:
    - ruta_ejecucion: ruta a la carpeta con los resultados.
    
    guarda cada matriz como 'cm_{nombre}.png' y las muestra.
    """
    top2 = pd.read_csv(os.path.join(ruta_ejecucion, "resumen_top2.csv"))
    for _, fila in top2.iterrows():
        cm = np.array(ast.literal_eval(fila["matriz_confusion"]))
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"matriz de confusion — {fila['nombre']}")
        plt.xlabel("prediccion")
        plt.ylabel("real")
        plt.colorbar()
        plt.tight_layout()
        out = os.path.join(ruta_ejecucion, f"cm_{fila['nombre']}.png")
        plt.savefig(out, dpi=150)
        print("guardado:", out)
        plt.show()

if __name__ == "__main__":
    ruta_in = input("ruta de ejecucion (carpeta con csvs) [enter=ultima]: ")
    ruta = normalizar_ruta(ruta_in)
    if not ruta or not os.path.isdir(ruta):
        print("no se encontro la carpeta de ejecucion. verifica que existan resultados en 'resultados/'.")
        raise SystemExit(1)
    faltan = verificar_csvs(ruta)
    if faltan:
        print("faltan archivos en la carpeta:", ", ".join(faltan))
        print("ruta usada:", ruta)
        raise SystemExit(1)
    curvas_entrenamiento(ruta)
    matrices_confusion(ruta)
