"""
este modulo contiene la logica para entrenar modelos de machine learning usando
descenso de gradiente estocastico con busqueda de hiperparametros y eliminacion
de candidatos durante el entrenamiento. se enfoca en modelos de regresion logistica
y svm lineal.
"""

from __future__ import annotations
import os, json, time, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Any, List, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier
from .datos import leer_tabla, preparar_Xy

SEMILLA = 42

@dataclass
class Candidato:
    """
    representa un candidato de modelo en la busqueda de hiperparametros.
    
    atributos:
    - nombre: identificador unico del candidato.
    - tipo: tipo de modelo ('regresion_logistica' o 'svm').
    - parametros: diccionario con los hiperparametros del modelo.
    - modelo: instancia del modelo sklearn (inicialmente None).
    - historial: lista de diccionarios con el historial de entrenamiento por epoca.
    """
    nombre: str
    tipo: str
    parametros: Dict[str, Any]
    modelo: Any = None
    historial: list = None

def _mapear_ritmo(s: str):
    """
    mapea el ritmo de aprendizaje a un valor valido para sklearn.
    
    parametros:
    - s: cadena con el ritmo deseado.
    
    retorna:
    - ritmo mapeado o 'optimal' si no es valido.
    """
    return s if s in ("optimal","constant","invscaling","adaptive") else "optimal"

def _construir_modelo(c: Candidato):
    """
    construye un modelo SGDClassifier basado en el tipo y parametros del candidato.
    
    parametros:
    - c: instancia de Candidato con tipo y parametros.
    
    retorna:
    - modelo configurado.
    
    lanza:
    - ValueError si el tipo no es soportado.
    """
    if c.tipo == "regresion_logistica":
        return SGDClassifier(
            loss="log_loss",
            penalty=c.parametros.get("penalizacion","l2"),
            alpha=c.parametros.get("alpha",1e-4),
            learning_rate=_mapear_ritmo(c.parametros.get("ritmo_aprendizaje","optimal")),
            eta0=c.parametros.get("eta0",0.0),
            power_t=c.parametros.get("power_t",0.5),
            max_iter=1, tol=None, random_state=SEMILLA
        )
    elif c.tipo == "svm":
        return SGDClassifier(
            loss=c.parametros.get("perdida","hinge"),
            penalty=c.parametros.get("penalizacion","l2"),
            alpha=c.parametros.get("alpha",1e-4),
            learning_rate=_mapear_ritmo(c.parametros.get("ritmo_aprendizaje","optimal")),
            eta0=c.parametros.get("eta0",0.0),
            power_t=c.parametros.get("power_t",0.5),
            max_iter=1, tol=None, random_state=SEMILLA
        )
    else:
        raise ValueError("Tipo de modelo no soportado")

def _iter_lotes(X, y, tamanio, rng):
    """
    genera lotes aleatorios de los datos para entrenamiento incremental.
    
    parametros:
    - X: matriz de caracteristicas.
    - y: vector de etiquetas.
    - tamanio: tamanio del lote.
    - rng: generador de numeros aleatorios.
    
    produce:
    - tuplas (X_lote, y_lote) para cada lote.
    """
    n = X.shape[0]
    idx = rng.permutation(n)
    for i in range(0, n, tamanio):
        sel = idx[i:i+tamanio]
        yield X[sel], y[sel]

def ejecutar(ruta_config: str):
    """
    ejecuta el proceso completo de entrenamiento con busqueda de hiperparametros.
    
    parametros:
    - ruta_config: ruta al archivo de configuracion json.
    
    retorna:
    - ruta del directorio de salida con resultados.
    
    el proceso incluye:
    - carga de configuracion y datos.
    - preparacion de datos (division, escalado).
    - creacion de candidatos de modelos.
    - entrenamiento por epocas con eliminacion de peores candidatos.
    - evaluacion final de los mejores candidatos en conjunto de prueba.
    - guardado de resultados en archivos csv y json.
    """
    with open(ruta_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dir_cfg = os.path.dirname(os.path.abspath(ruta_config))
    ruta_datos = cfg["datos"]["ruta"]
    if not os.path.isabs(ruta_datos):
        ruta_datos = os.path.abspath(os.path.join(dir_cfg, ruta_datos))

    raiz_salida = cfg["salida"].get("raiz","../resultados")
    if not os.path.isabs(raiz_salida):
        raiz_salida = os.path.abspath(os.path.join(dir_cfg, raiz_salida))
    run_dir = os.path.join(raiz_salida, time.strftime("ejecucion-%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    df = leer_tabla(ruta_datos)
    X, y, etiqueta_usada, mapa_etiquetas = preparar_Xy(df, cfg["datos"].get("etiqueta"))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["datos"].get("proporcion_prueba",0.2),
        random_state=cfg["datos"].get("semilla",42), stratify=y
    )
    if cfg["datos"].get("escalar", True):
        esc = StandardScaler(with_mean=False)
        Xtr = esc.fit_transform(Xtr)
        Xte = esc.transform(Xte)

    pool: List[Candidato] = []
    for d in cfg["busqueda"].get("regresion_logistica", []):
        pool.append(Candidato(nombre=d["nombre"], tipo="regresion_logistica", parametros=d, historial=[]))
    for d in cfg["busqueda"].get("svm", []):
        pool.append(Candidato(nombre=d["nombre"], tipo="svm", parametros=d, historial=[]))

    rng = np.random.RandomState(SEMILLA)
    clases = np.unique(ytr)

    for c in pool:
        c.modelo = _construir_modelo(c)
        lote0 = max(32, c.parametros.get("tamanio_lote",256))
        xb, yb = next(_iter_lotes(Xtr, ytr, lote0, rng))
        c.modelo.partial_fit(xb, yb, classes=clases)

    epocas = cfg["entrenamiento"].get("epocas",20)
    eliminar_cada = cfg["entrenamiento"].get("eliminar_cada",5)
    limite_eval = cfg["entrenamiento"].get("limite_eval_entrenamiento",0)

    filas_log = []
    for ep in range(1, epocas+1):
        for c in pool:
            bs = c.parametros.get("tamanio_lote",256)
            for xb, yb in _iter_lotes(Xtr, ytr, bs, rng):
                c.modelo.partial_fit(xb, yb)

        if limite_eval and Xtr.shape[0] > limite_eval:
            Xev, yev = Xtr[:limite_eval], ytr[:limite_eval]
        else:
            Xev, yev = Xtr, ytr

        for c in pool:
            pred = c.modelo.predict(Xev)
            acc = accuracy_score(yev, pred)
            c.historial.append({"epoca": ep, "exactitud_entrenamiento": acc})
            filas_log.append({"epoca": ep, "nombre": c.nombre, "tipo": c.tipo, "exactitud_entrenamiento": acc})

        if (ep % eliminar_cada == 0) and len(pool) > 2:
            peor_idx = min([(c.historial[-1]["exactitud_entrenamiento"], i) for i, c in enumerate(pool)])[1]
            eliminado = pool.pop(peor_idx)
            print(f"[epoca {ep}] descartado → {eliminado.nombre} ({eliminado.tipo})")

    pool_ordenado = sorted(pool, key=lambda c: c.historial[-1]["exactitud_entrenamiento"], reverse=True)
    finalistas = pool_ordenado[:2]

    filas_test = []
    for c in finalistas:
        y_pred = c.modelo.predict(Xte)
        acc = accuracy_score(yte, y_pred)
        rep = classification_report(yte, y_pred, output_dict=False)
        cm  = confusion_matrix(yte, y_pred).tolist()
        filas_test.append({"nombre": c.nombre, "tipo": c.tipo, "exactitud_prueba": acc,
                           "informe_clasificacion": rep, "matriz_confusion": cm})
        print(f"\n=== {c.nombre} ({c.tipo}) ===")
        print(f"exactitud en prueba: {acc:.4f}")
        print(rep)

    pd.DataFrame(filas_log).to_csv(os.path.join(run_dir,"registro_entrenamiento.csv"), index=False)
    pd.DataFrame(filas_test).to_csv(os.path.join(run_dir,"resumen_top2.csv"), index=False)
    meta = {
        "etiqueta_usada": etiqueta_usada,
        "candidatos_ordenados": [c.nombre for c in pool_ordenado],
        "mapa_etiquetas": {str(k): v for k, v in (mapa_etiquetas or {}).items()},
    }
    with open(os.path.join(run_dir,"meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return run_dir
