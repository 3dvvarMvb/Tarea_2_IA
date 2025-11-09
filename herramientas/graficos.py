import os, ast, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detectar_ultima():
    base = "resultados"
    if not os.path.isdir(base):
        return None
    dirs = [d for d in os.listdir(base) if d.startswith("ejecucion-")]
    if not dirs:
        return None
    dirs.sort()
    return os.path.join(base, dirs[-1])

def normalizar_ruta(ruta: str):
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
    faltan = []
    for nombre in ("registro_entrenamiento.csv", "resumen_top2.csv"):
        if not os.path.isfile(os.path.join(ruta, nombre)):
            faltan.append(nombre)
    return faltan

def curvas_entrenamiento(ruta_ejecucion: str):
    log = pd.read_csv(os.path.join(ruta_ejecucion, "registro_entrenamiento.csv"))
    plt.figure()
    for nombre in sorted(log['nombre'].unique()):
        sub = log[log['nombre'] == nombre]
        plt.plot(sub['epoca'], sub['exactitud_entrenamiento'], label=nombre)
    plt.xlabel("Época")
    plt.ylabel("Exactitud (entrenamiento)")
    plt.title("Evolución de candidatos")
    plt.grid(True)
    plt.legend(loc="best")
    out = os.path.join(ruta_ejecucion, "curva_entrenamiento.png")
    plt.savefig(out, dpi=150)
    print("Guardado:", out)
    plt.show()

def matrices_confusion(ruta_ejecucion: str):
    top2 = pd.read_csv(os.path.join(ruta_ejecucion, "resumen_top2.csv"))
    for _, fila in top2.iterrows():
        cm = np.array(ast.literal_eval(fila["matriz_confusion"]))
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(f"Matriz de confusión — {fila['nombre']}")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.colorbar()
        plt.tight_layout()
        out = os.path.join(ruta_ejecucion, f"cm_{fila['nombre']}.png")
        plt.savefig(out, dpi=150)
        print("Guardado:", out)
        plt.show()

if __name__ == "__main__":
    ruta_in = input("Ruta de ejecución (carpeta con CSVs) [ENTER=última]: ")
    ruta = normalizar_ruta(ruta_in)
    if not ruta or not os.path.isdir(ruta):
        print("No se encontró la carpeta de ejecución. Verifica que existan resultados en 'resultados/'.")
        raise SystemExit(1)
    faltan = verificar_csvs(ruta)
    if faltan:
        print("Faltan archivos en la carpeta:", ", ".join(faltan))
        print("Ruta usada:", ruta)
        raise SystemExit(1)
    curvas_entrenamiento(ruta)
    matrices_confusion(ruta)
