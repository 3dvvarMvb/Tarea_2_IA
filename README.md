<<<<<<< HEAD
# Tarea2IAParte2
Parte 2 de la tarea 2 de Inteligencia Artificial
=======
# Tarea 2 — Parte 2 (Clasificación | Listo para VS Code/Jupyter)

Repositorio con la **implementación de la Parte 2**: entrenamiento supervisado con eliminación periódica y selección de los 2 mejores modelos. Pensado para correr sin fricción en **VS Code** o **Jupyter**.

## Carpeta y archivos clave
```
Tarea2/
├─ configuraciones/
│  └─ parte2.json
├─ datos/
│  └─ Dry_Bean_Dataset.xlsx      # ← coloca aquí tu dataset (XLSX o CSV)
├─ cuadernos/
│  └─ Parte2_Solo.ipynb
├─ aplicacion/
│  ├─ datos.py
│  └─ entrenamiento.py
├─ herramientas/
│  └─ graficos.py
├─ resultados/
├─ EjecutarParte2.py
└─ requerimientos.txt
```

## Pasos para usarlo

### 1) Dataset
- Copia el archivo en `datos/` (por defecto se usa `Dry_Bean_Dataset.xlsx`).
- Abre `configuraciones/parte2.json` y ajusta:
  - `datos.ruta`: ruta al XLSX/CSV (ej. `../datos/Dry_Bean_Dataset.xlsx`)
  - `datos.etiqueta`: nombre de la columna objetivo (ej. `Class`)
  - `proporcion_prueba`, `semilla`, `escalar` según necesites

### 2) Dependencias
```bash
cd Tarea2
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requerimientos.txt
```

### 3) Ejecutar
- **Terminal (recomendado):**
  ```bash
  python EjecutarParte2.py --config configuraciones/parte2.json
  ```
- **Jupyter (alternativa):**
  - Abre `cuadernos/Parte2_Solo.ipynb`
  - Selecciona el kernel de tu `.venv` y ejecuta la celda

## Salidas esperadas
Se crea una carpeta `resultados/ejecucion-YYYYmmdd-HHMMSS/` con:
- `registro_entrenamiento.csv` — historial por época (exactitud de entrenamiento por candidato)
- `resumen_top2.csv` — métricas finales de test de los 2 mejores (accuracy, reporte, matriz de confusión)
- `meta.json` — etiqueta usada y orden de candidatos

Para generar imágenes (curva y matrices de confusión):
```bash
python herramientas/graficos.py
# cuando pida la ruta: resultados/ejecucion-AAAAmmdd-HHMMSS
```

## Ajustes rápidos (en `parte2.json`)
- `entrenamiento.epocas` — número de épocas totales
- `entrenamiento.eliminar_cada` — cada cuántas épocas se descarta el peor
- `busqueda.regresion_logistica` / `busqueda.svm` — ≥3 configs por técnica (alpha, ritmo de aprendizaje, tamaño de lote, etc.)

## Requisitos
- Python 3.x
- Paquetes en `requerimientos.txt` (incluye `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`)
>>>>>>> 1a542c3 (Tarea2 Parte 2 Primer commit)
