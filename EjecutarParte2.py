import os, argparse
from aplicacion.entrenamiento import ejecutar

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ejecuta la Parte 2 (entrenamiento supervisado en paralelo con eliminación).")
    ap.add_argument("--config", type=str, default="configuraciones/parte2.json", help="Ruta al archivo de configuración (JSON).")
    args = ap.parse_args()
    carpeta = ejecutar(os.path.abspath(args.config))
    print("\nArtefactos generados en:", carpeta)
