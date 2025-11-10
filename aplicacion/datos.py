"""
este modulo contiene funciones para cargar y preparar datos para tareas de machine learning,
incluyendo lectura de tablas, preparacion de caracteristicas y etiquetas, y configuracion
para clustering.
"""

import os
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def leer_tabla(ruta: str) -> pd.DataFrame:
    """
    lee una tabla desde un archivo y la devuelve como dataframe de pandas.
    
    soporta formatos csv, txt, xlsx y xls. para csv y txt, intenta primero con
    separador ';' y decimal ',', si no funciona, usa los valores por defecto.
    
    parametros:
    - ruta: ruta al archivo de datos.
    
    retorna:
    - dataframe con los datos.
    
    lanza:
    - FileNotFoundError si el archivo no existe.
    - ValueError si el formato no es soportado.
    """
    ruta = os.path.abspath(ruta)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"no se encontro el dataset en: {ruta}")
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(ruta, sep=";", decimal=",")
        if df.shape[1] == 1:
            df = pd.read_csv(ruta)
        return df
    else:
        raise ValueError(f"formato de archivo no soportado: {ext}")

def preparar_Xy(df: pd.DataFrame, columna_etiqueta: str | None):
    """
    prepara las caracteristicas X y las etiquetas y desde un dataframe.
    
    infiere la columna de etiqueta si no se proporciona, buscando nombres comunes.
    convierte columnas no numericas a numericas, rellena valores faltantes con medianas,
    y codifica etiquetas categoricas a enteros.
    
    parametros:
    - df: dataframe con los datos.
    - columna_etiqueta: nombre de la columna de etiquetas, o None para inferir.
    
    retorna:
    - X: matriz de caracteristicas como numpy array.
    - y: vector de etiquetas como numpy array de enteros.
    - columna_etiqueta: nombre de la columna usada como etiqueta.
    """
    if not columna_etiqueta:
        candidatos = [c for c in df.columns if c.lower() in ("class","label","target","y")]
        columna_etiqueta = candidatos[0] if candidatos else df.columns[-1]

    y_raw = df[columna_etiqueta]
    X_df = df.drop(columns=[columna_etiqueta])

    for c in X_df.select_dtypes(include=["object","bool"]).columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    if y_raw.dtype.kind in "OUSb":
        y = pd.Categorical(y_raw).codes + 1
    else:
        y = y_raw.to_numpy().astype(int)

    X = X_df.to_numpy(dtype="float64")
    return X, y.astype(int), columna_etiqueta

def preparar_datos_clustering(
    ruta_datos: str,
    columna_etiqueta: str | None = None,
    proporcion_prueba: float = 0.2,
    semilla: int = 42,
    escalar: bool = True,
) -> Dict[str, Any]:
    """
    genera los conjuntos de entrenamiento y prueba listos para tareas de clustering.
    
    carga los datos, prepara caracteristicas y etiquetas, divide en train/test,
    y opcionalmente escala las caracteristicas.
    
    parametros:
    - ruta_datos: ruta al archivo de datos.
    - columna_etiqueta: nombre de la columna de etiquetas, o None para inferir.
    - proporcion_prueba: fraccion de datos para prueba (0.0 a 1.0).
    - semilla: semilla para division aleatoria.
    - escalar: si escalar las caracteristicas con StandardScaler.
    
    retorna:
    - diccionario con X_entrenamiento, X_prueba, y_entrenamiento, y_prueba,
      etiqueta, mapa_etiquetas, escalador, df_features, y_completa.
    """
    df = leer_tabla(ruta_datos)
    if not columna_etiqueta:
        candidatos = [c for c in df.columns if c.lower() in ("class", "label", "target", "y")]
        columna_etiqueta = candidatos[0] if candidatos else df.columns[-1]

    y_bruta = pd.Categorical(df[columna_etiqueta])
    y_series = pd.Series(y_bruta.codes, name=columna_etiqueta, index=df.index)
    mapa_etiquetas = {int(code): str(label) for code, label in enumerate(y_bruta.categories)}

    X_df = df.drop(columns=[columna_etiqueta]).copy()
    for c in X_df.select_dtypes(include=["object", "bool"]).columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    escalador = None
    if escalar:
        escalador = StandardScaler()
        X_df = pd.DataFrame(
            escalador.fit_transform(X_df),
            columns=X_df.columns,
            index=X_df.index,
        )

    X_ent, X_pru, y_ent, y_pru = train_test_split(
        X_df,
        y_series,
        test_size=proporcion_prueba,
        random_state=semilla,
        stratify=y_series,
    )

    return {
        "X_entrenamiento": X_ent,
        "X_prueba": X_pru,
        "y_entrenamiento": y_ent,
        "y_prueba": y_pru,
        "etiqueta": columna_etiqueta,
        "mapa_etiquetas": mapa_etiquetas,
        "escalador": escalador,
        "df_features": X_df,
        "y_completa": y_series,
    }
