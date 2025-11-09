import os
import pandas as pd
import numpy as np

def leer_tabla(ruta: str) -> pd.DataFrame:
    ruta = os.path.abspath(ruta)
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta}")
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(ruta)
    elif ext in (".csv", ".txt"):
        return pd.read_csv(ruta)
    else:
        raise ValueError(f"Formato de archivo no soportado: {ext}")

def preparar_Xy(df: pd.DataFrame, columna_etiqueta: str | None):
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
