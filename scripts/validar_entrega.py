#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validar_entrega.py — Corrector automático del SIMULACRO de Evaluación Parcial
=============================================================================
Mini-proyecto "AutoAndes Import" (dataset mpg) · Bloque 1 (S01–S07)

Uso:
    python validar_entrega.py <carpeta_o_zip_de_entrega>
    python validar_entrega.py entrega.zip --inicio "2026-08-14 08:00" \
        --fin "2026-08-14 10:00" --correo alumno@pucp.edu.pe

Qué hace:
  1. Si recibe un .zip lo descomprime a una carpeta temporal y localiza la raíz
     del proyecto (la carpeta que contiene eda_autos.ipynb o .git).
  2. RECOMPUTA la referencia desde 'mpg' (seaborn -> fallback al mpg.csv que
     vive junto a este script) con la misma limpieza del enunciado (dropna) y
     los mismos 3 modelos (seed 42, test 0.2).  No hardcodea cifras.
  3. Verifica cada ítem de la rúbrica imprimiendo [OK]/[FALLA] y asignando
     puntos parciales:
        P1 Entorno (2), P2 DVC (2), P3 EDA (4), P4 Git (2), P5 GitHub (1),
        P6 MLflow (3), P7 App (2), P8 Teoría (4), Bonus Registry (+1).
  4. Emite [ALERTA] (sin puntaje) por señales de autoría: commits fuera de la
     ventana del examen, correo del autor distinto al declarado, e
     inconsistencias entre hallazgos.json y mlruns/.
  5. Imprime la NOTA final (tope 20) y termina con exit 0 si la entrega pudo
     procesarse, o exit 1 si la estructura es ilegible (sin carpeta/zip válido).

Tolerancias: TOL = 0.01 (rel/abs, hallazgos del EDA);
             TOL_MODELOS = 0.05 (métricas de modelos: absorbe variaciones
             menores entre versiones de scikit-learn).

Corrección por lotes:
    for z in entregas/*.zip; do
        python validar_entrega.py "$z" > "reportes/$(basename "$z" .zip).txt"
    done
"""

import argparse
import ast
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime

# --------------------------------------------------------------------------- #
# Configuración del simulacro (contrato del enunciado)
# --------------------------------------------------------------------------- #
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))

TOL = 0.01           # hallazgos del EDA (math.isclose rel/abs)
TOL_MODELOS = 0.05   # métricas de modelos (variación entre versiones sklearn)

NOTEBOOK = "eda_autos.ipynb"
KERNEL = "datalab"
DATA_CRUDA = "autos.csv"
DATA_LIMPIA = "autos_limpio.csv"
EXPERIMENTO = "prediccion-eficiencia-autos"
MODELO_REGISTRADO = "eficiencia-autos"
RAMA_FEATURE = "feature-eda"
RUNS_REQUERIDOS = ["baseline-lineal", "rf-100-5", "rf-300-10"]
FIGURAS = ["scatter_weight_mpg.png", "mpg_por_origen.png"]
METRICAS = ["rmse_test", "mae_test", "r2_test"]
PAQUETES = ["pandas", "seaborn", "scikit-learn", "jupyterlab",
            "ipykernel", "mlflow", "dvc", "streamlit"]

# Clave de teoría del simulacro (coincide con el deck simulacro_parcial.pdf).
RESPUESTAS = {"p1": "b", "p2": "a", "p3": "c", "p4": "b",
              "p5": "a", "p6": "c", "p7": "a", "p8": "b"}

# Fallback si scikit-learn no estuviera disponible al corregir (recomputado
# el 2026-08-14 con seed 42 / test 0.2; solo se usa como último recurso).
REF_MODELOS_FALLBACK = {
    "baseline-lineal": {"rmse_test": 3.2407, "mae_test": 2.5039, "r2_test": 0.7942},
    "rf-100-5":        {"rmse_test": 2.5591, "mae_test": 1.8897, "r2_test": 0.8717},
    "rf-300-10":       {"rmse_test": 2.3975, "mae_test": 1.7004, "r2_test": 0.8874},
}

# --------------------------------------------------------------------------- #
# Reporte y puntaje
# --------------------------------------------------------------------------- #
_puntajes = []   # [(item, obtenido, maximo)]
_alertas = []


def check(cond, desc):
    print("{} {}".format("[OK]  " if cond else "[FALLA]", desc))
    return bool(cond)


def alerta(desc):
    print("[ALERTA] {}".format(desc))
    _alertas.append(desc)


def puntuar(item, obtenido, maximo):
    obtenido = max(0.0, min(float(obtenido), float(maximo)))
    _puntajes.append((item, obtenido, maximo))
    print("   >> {}: {:.2f} / {:.2f} pt".format(item, obtenido, maximo))


def cerca(v_est, v_ref, tol=TOL):
    try:
        return math.isclose(float(v_est), float(v_ref), rel_tol=tol, abs_tol=tol)
    except (TypeError, ValueError):
        return False


def seccion(titulo):
    print("\n-- {} --".format(titulo))


# --------------------------------------------------------------------------- #
# Referencia recomputada (misma lógica que el enunciado)
# --------------------------------------------------------------------------- #
def cargar_dataset():
    """seaborn.load_dataset('mpg'); si falla, el mpg.csv junto a este script."""
    import pandas as pd
    try:
        import seaborn as sns
        df = sns.load_dataset("mpg")
        if "mpg" in df.columns:
            return df
    except Exception:
        pass
    return pd.read_csv(os.path.join(DIR_SCRIPT, "mpg.csv"))


def calcular_referencia():
    df = cargar_dataset()
    ref = {
        "n_filas_original": int(df.shape[0]),
        "n_columnas": int(df.shape[1]),
        "nulos_horsepower": int(df["horsepower"].isna().sum()),
    }
    limpio = df.dropna().copy()
    ref["n_filas_limpio"] = int(limpio.shape[0])
    ref["mpg_promedio"] = round(float(limpio["mpg"].mean()), 4)
    ref["mpg_mediano"] = round(float(limpio["mpg"].median()), 4)
    ref["correlacion_weight_mpg"] = round(float(limpio["weight"].corr(limpio["mpg"])), 4)
    por_origen = limpio.groupby("origin")["mpg"].mean()
    ref["mpg_promedio_por_origen"] = {k: round(float(v), 4) for k, v in por_origen.items()}
    ref["origen_mas_eficiente"] = str(por_origen.idxmax())
    ref["modelos"] = calcular_referencia_modelos(limpio)
    return ref


def calcular_referencia_modelos(limpio):
    """Recomputa las métricas de los 3 runs; fallback a constantes si no hay sklearn."""
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                     r2_score)
        from sklearn.model_selection import train_test_split
    except Exception:
        print("[INFO] scikit-learn no disponible: uso métricas de referencia precalculadas.")
        return dict(REF_MODELOS_FALLBACK)
    cols = ["cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model_year"]
    X, y = limpio[cols], limpio["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    modelos = {
        "baseline-lineal": LinearRegression(),
        "rf-100-5": RandomForestRegressor(n_estimators=100, max_depth=5,
                                          random_state=42, n_jobs=-1),
        "rf-300-10": RandomForestRegressor(n_estimators=300, max_depth=10,
                                           random_state=42, n_jobs=-1),
    }
    ref = {}
    for nombre, m in modelos.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        ref[nombre] = {
            "rmse_test": round(float(np.sqrt(mean_squared_error(y_te, pred))), 4),
            "mae_test": round(float(mean_absolute_error(y_te, pred)), 4),
            "r2_test": round(float(r2_score(y_te, pred)), 4),
        }
    return ref


# --------------------------------------------------------------------------- #
# Localización de la entrega (carpeta o zip)
# --------------------------------------------------------------------------- #
def localizar_raiz(ruta):
    """Devuelve (raiz, tmpdir|None). Busca la carpeta con el notebook o .git."""
    tmpdir = None
    if os.path.isfile(ruta) and ruta.lower().endswith(".zip"):
        tmpdir = tempfile.mkdtemp(prefix="entrega_")
        with zipfile.ZipFile(ruta) as zf:
            zf.extractall(tmpdir)
        ruta = tmpdir
    if not os.path.isdir(ruta):
        return None, tmpdir
    con_notebook, con_git = [], []
    for dirpath, dirnames, filenames in os.walk(ruta):
        if ".git" in dirnames:
            con_git.append(dirpath)
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "mlruns", ".dvc", "__pycache__"}]
        if NOTEBOOK in filenames:
            con_notebook.append(dirpath)
    for lista in (con_notebook, con_git):
        if lista:
            lista.sort(key=lambda p: p.count(os.sep))
            return lista[0], tmpdir
    return None, tmpdir


# --------------------------------------------------------------------------- #
# P1 — Entorno (S01) · 2 pt
# --------------------------------------------------------------------------- #
def validar_entorno(raiz):
    seccion("P1 Entorno (S01)")
    pts = 0.0
    contenido = ""
    for i, nombre in enumerate(["environment.yml", "environment_hist.yml"]):
        rutayml = os.path.join(raiz, nombre)
        existe = os.path.isfile(rutayml)
        if check(existe, "existe {}".format(nombre)) :
            pts += 0.2
            try:
                with open(rutayml, encoding="utf-8", errors="replace") as f:
                    contenido += f.read().lower() + "\n"
            except OSError:
                pass
    tiene_python = bool(re.search(r"python\s*[>=<]*\s*3\.11", contenido))
    if check(tiene_python, "declara python=3.11"):
        pts += 0.4
    for paq in PAQUETES:
        patron = re.escape(paq.lower())
        presente = bool(re.search(r"(^|[\s\-'\"]){}([=<>\s'\",]|$)".format(patron),
                                  contenido, re.MULTILINE))
        if check(presente, "paquete declarado: {}".format(paq)):
            pts += 0.15
    puntuar("P1 Entorno", pts, 2.0)


# --------------------------------------------------------------------------- #
# P2 — DVC (S06) · 2 pt
# --------------------------------------------------------------------------- #
def md5_de(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_puntero_dvc(ruta):
    """Extrae (md5, path, hash_moderno) de un puntero .dvc (YAML simple)."""
    try:
        with open(ruta, encoding="utf-8", errors="replace") as f:
            texto = f.read()
    except OSError:
        return None, None, False
    m_md5 = re.search(r"md5:\s*([0-9a-f]{32})", texto)
    m_path = re.search(r"path:\s*(\S+)", texto)
    return (m_md5.group(1) if m_md5 else None,
            m_path.group(1) if m_path else None,
            "hash: md5" in texto)


def validar_dvc(raiz, ref):
    seccion("P2 DVC (S06)")
    pts = 0.0
    cfg = os.path.join(raiz, ".dvc", "config")
    if check(os.path.isfile(cfg), "existe .dvc/config (dvc init)"):
        pts += 0.3
        with open(cfg, encoding="utf-8", errors="replace") as f:
            texto_cfg = f.read()
        if check("almacen" in texto_cfg, "remoto 'almacen' configurado"):
            pts += 0.2
    else:
        check(False, "remoto 'almacen' configurado")
    if check(os.path.isfile(os.path.join(raiz, ".dvcignore")), "existe .dvcignore"):
        pts += 0.1
    for csv in [DATA_CRUDA, DATA_LIMPIA]:
        puntero = os.path.join(raiz, "data", csv + ".dvc")
        datocsv = os.path.join(raiz, "data", csv)
        md5_decl, _, hash_moderno = leer_puntero_dvc(puntero)
        ok = False
        if md5_decl and os.path.isfile(datocsv):
            ok = (md5_decl == md5_de(datocsv))
            if not ok and not hash_moderno:
                # dvc < 3 hashea archivos de texto normalizando CRLF -> LF
                with open(datocsv, "rb") as f:
                    normalizado = f.read().replace(b"\r\n", b"\n")
                ok = (md5_decl == hashlib.md5(normalizado).hexdigest())
        if check(ok, "md5 de data/{}.dvc == hash recalculado del CSV".format(csv)):
            pts += 0.5
    limpio = os.path.join(raiz, "data", DATA_LIMPIA)
    filas_ok = False
    if os.path.isfile(limpio):
        try:
            import pandas as pd
            filas_ok = (len(pd.read_csv(limpio)) == ref["n_filas_limpio"])
        except Exception:
            filas_ok = False
    if check(filas_ok, "data/{} con {} filas (limpieza del enunciado)".format(
            DATA_LIMPIA, ref["n_filas_limpio"])):
        pts += 0.4
    puntuar("P2 DVC", pts, 2.0)


# --------------------------------------------------------------------------- #
# P3 — EDA (S02) · 4 pt
# --------------------------------------------------------------------------- #
def validar_eda(raiz, ref):
    seccion("P3 EDA (S02)")
    pts = 0.0
    # (a) kernel + celdas ejecutadas
    ruta_nb = os.path.join(raiz, NOTEBOOK)
    kernel, ejecutadas = None, 0
    if os.path.isfile(ruta_nb):
        try:
            with open(ruta_nb, encoding="utf-8-sig") as f:
                nb = json.load(f)
            if not isinstance(nb, dict):
                nb = {}
            kernel = nb.get("metadata", {}).get("kernelspec", {}).get("name")
            ejecutadas = sum(1 for c in nb.get("cells", [])
                             if isinstance(c, dict) and c.get("cell_type") == "code"
                             and c.get("execution_count") is not None)
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    if check(kernel == KERNEL,
             "kernelspec.name del notebook == '{}' (encontrado: {})".format(KERNEL, kernel)):
        pts += 0.5
    if check(ejecutadas >= 5,
             ">= 5 celdas de código ejecutadas (encontradas: {})".format(ejecutadas)):
        pts += 0.5

    # (b) hallazgos.json contra la referencia recomputada
    ruta_h = os.path.join(raiz, "resultados", "hallazgos.json")
    est = None
    if os.path.isfile(ruta_h):
        try:
            with open(ruta_h, encoding="utf-8-sig") as f:
                est = json.load(f)
            if not isinstance(est, dict):
                est = None
        except (json.JSONDecodeError, OSError, ValueError):
            est = None
    if not check(est is not None, "resultados/hallazgos.json existe y es JSON válido"):
        est = {}
    comprobaciones = []
    for k in ["n_filas_original", "n_columnas", "nulos_horsepower",
              "n_filas_limpio", "mpg_promedio", "mpg_mediano",
              "correlacion_weight_mpg"]:
        comprobaciones.append((cerca(est.get(k), ref[k]),
                               "'{}' = {} (ref {})".format(k, est.get(k), ref[k])))
    sub = est.get("mpg_promedio_por_origen", {})
    for origen, vref in ref["mpg_promedio_por_origen"].items():
        v = sub.get(origen) if isinstance(sub, dict) else None
        comprobaciones.append((cerca(v, vref),
                               "mpg_promedio_por_origen[{}] = {} (ref {})".format(origen, v, vref)))
    comprobaciones.append((est.get("origen_mas_eficiente") == ref["origen_mas_eficiente"],
                           "origen_mas_eficiente == '{}' (dado: {})".format(
                               ref["origen_mas_eficiente"], est.get("origen_mas_eficiente"))))
    mejor_ref = min(ref["modelos"], key=lambda r: ref["modelos"][r]["rmse_test"])
    comprobaciones.append((est.get("mejor_modelo") == mejor_ref,
                           "mejor_modelo == '{}' (dado: {})".format(mejor_ref, est.get("mejor_modelo"))))
    for k, mk in [("rmse_test_mejor_modelo", "rmse_test"),
                  ("r2_test_mejor_modelo", "r2_test")]:
        vref = ref["modelos"][mejor_ref][mk]
        comprobaciones.append((cerca(est.get(k), vref, TOL_MODELOS),
                               "'{}' = {} (ref {}, tol {})".format(k, est.get(k), vref, TOL_MODELOS)))
    aciertos = 0
    for ok, desc in comprobaciones:
        if check(ok, desc):
            aciertos += 1
    pts += 2.5 * aciertos / len(comprobaciones)

    # (c) figuras
    for fig in FIGURAS:
        ruta_fig = os.path.join(raiz, "resultados", "figuras", fig)
        if check(os.path.isfile(ruta_fig) and os.path.getsize(ruta_fig) > 0,
                 "figura resultados/figuras/{} presente".format(fig)):
            pts += 0.25
    puntuar("P3 EDA", pts, 4.0)
    return est


# --------------------------------------------------------------------------- #
# P4 / P5 — Git y GitHub (S03 + S04) · 2 + 1 pt
# --------------------------------------------------------------------------- #
def git(raiz, *args):
    try:
        r = subprocess.run(["git", "-C", raiz] + list(args),
                           capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip()
    except Exception:
        return 1, ""


def validar_git(raiz, inicio, fin, correo):
    seccion("P4 Git (S03)")
    pts = 0.0
    if not os.path.isdir(os.path.join(raiz, ".git")):
        check(False, "carpeta .git presente en la entrega")
        puntuar("P4 Git", 0.0, 2.0)
        return
    check(True, "carpeta .git presente en la entrega")
    cod, log = git(raiz, "log", "--all", "--pretty=format:%H|%ae|%at|%s")
    commits = [l.split("|", 3) for l in log.splitlines() if l.count("|") >= 3] if cod == 0 else []
    validos = [c for c in commits if len(c[3].strip()) >= 10]
    if check(len(validos) >= 6,
             ">= 6 commits con mensajes >= 10 caracteres (válidos: {}/{})".format(
                 len(validos), len(commits))):
        pts += 1.0
    else:
        pts += 1.0 * min(1.0, len(validos) / 6.0)
    principal = "main"
    cod_m, _ = git(raiz, "rev-parse", "--verify", "main")
    if cod_m != 0:
        principal = "master"
    _, ramas = git(raiz, "branch", "--list", RAMA_FEATURE)
    _, remotas_git = git(raiz, "branch", "-r")
    tiene_rama = (RAMA_FEATURE in ramas) or \
                 ("origin/{}".format(RAMA_FEATURE) in remotas_git)
    cod_a, _ = git(raiz, "merge-base", "--is-ancestor", RAMA_FEATURE, principal)
    _, msgs_merge = git(raiz, "log", "--merges", "--pretty=%s", principal)
    fusionada = (tiene_rama and cod_a == 0) or (RAMA_FEATURE in msgs_merge)
    if check(fusionada,
             "rama '{}' fusionada en {} (rama o commit de fusión presentes)".format(
                 RAMA_FEATURE, principal)):
        pts += 1.0
    elif tiene_rama:
        pts += 0.5
    puntuar("P4 Git", pts, 2.0)

    # Alertas de autoría (sin puntaje).
    correos = sorted({c[1] for c in commits})
    if correo and correos and any(c != correo for c in correos):
        alerta("correos de autor {} distintos del declarado {}".format(correos, correo))
    elif len(correos) > 1:
        alerta("más de un correo de autor en los commits: {}".format(correos))
    if commits:
        fechas = sorted(datetime.fromtimestamp(int(c[2])) for c in commits)
        print("[INFO] commits entre {} y {}".format(fechas[0], fechas[-1]))
        if inicio and fechas[0] < inicio:
            alerta("hay commits ANTERIORES al inicio del examen ({})".format(fechas[0]))
        if fin and fechas[-1] > fin:
            alerta("hay commits POSTERIORES al fin del examen ({})".format(fechas[-1]))
        if len(fechas) >= 2 and (fechas[-1] - fechas[0]).total_seconds() < 600:
            alerta("todos los commits caben en menos de 10 minutos: trabajo no repartido")

    seccion("P5 GitHub (S04)")
    pts5 = 0.0
    _, url = git(raiz, "config", "--get", "remote.origin.url")
    if check("github.com" in url, "remoto origin apunta a github.com ({})".format(url or "no configurado")):
        pts5 += 0.5
    tiene_fixes = any(re.search(r"(fix(e[sd])?|close[sd]?|resolve[sd]?)\s*#\d+",
                                c[3], re.IGNORECASE) for c in commits)
    if check(tiene_fixes, "existe un commit con 'Fixes #N' (cierre de issue, S04)"):
        pts5 += 0.25
    _, remotas = git(raiz, "branch", "-r")
    if check(bool(re.search(r"origin/(main|master)", remotas)),
             "referencia remota origin/main presente (evidencia de push)"):
        pts5 += 0.25
    puntuar("P5 GitHub", pts5, 1.0)


# --------------------------------------------------------------------------- #
# P6 — MLflow (S07) · 3 pt (+1 bonus Registry)
#
# MLflow >= 3.15 (el de S07) usa por defecto sqlite:///mlflow.db para los
# metadatos y mlruns/ para los artefactos.  El validador lee mlflow.db con
# sqlite3 (stdlib); si no existe, cae al file store antiguo (mlruns/meta.yaml)
# para entregas hechas con un MLflow anterior.
# --------------------------------------------------------------------------- #
def leer_runs_sqlite(db):
    """(exp_ok, runs, registrado, alias_ok) desde mlflow.db."""
    import sqlite3
    exp_ok, runs, registrado, alias_ok = False, {}, False, False
    try:
        con = sqlite3.connect(db)
        fila = con.execute("SELECT experiment_id FROM experiments WHERE name=?",
                           (EXPERIMENTO,)).fetchone()
        if fila:
            exp_ok = True
            for run_uuid, nombre in con.execute(
                    "SELECT run_uuid, name FROM runs "
                    "WHERE experiment_id=? AND lifecycle_stage='active'", (fila[0],)):
                params = dict(con.execute(
                    "SELECT key, value FROM params WHERE run_uuid=?", (run_uuid,)))
                metrics = {}
                for k, v in con.execute(
                        "SELECT key, value FROM metrics WHERE run_uuid=? "
                        "ORDER BY timestamp, step", (run_uuid,)):
                    metrics[k] = float(v)          # se queda el último valor
                previo = runs.get(nombre)
                if previo is None or len(metrics) > len(previo["metrics"]):
                    runs[nombre] = {"params": params, "metrics": metrics}
        registrado = con.execute(
            "SELECT 1 FROM registered_models WHERE name=?",
            (MODELO_REGISTRADO,)).fetchone() is not None
        alias_ok = con.execute(
            "SELECT 1 FROM registered_model_aliases WHERE name=? AND alias='champion'",
            (MODELO_REGISTRADO,)).fetchone() is not None
        con.close()
    except Exception as e:
        print("[INFO] no se pudo leer mlflow.db: {}".format(e))
    return exp_ok, runs, registrado, alias_ok


def leer_runs_filestore(raiz):
    """(exp_ok, runs, registrado, alias_ok) desde el file store antiguo mlruns/."""
    base = os.path.join(raiz, "mlruns")
    encontrados, exp_ok = {}, False
    if not os.path.isdir(base):
        return exp_ok, encontrados, False, False
    for exp_dir in glob.glob(os.path.join(base, "*")):
        meta = os.path.join(exp_dir, "meta.yaml")
        if not os.path.isfile(meta):
            continue
        with open(meta, encoding="utf-8", errors="replace") as f:
            texto = f.read()
        if not re.search(r"name:\s*{}\s*$".format(re.escape(EXPERIMENTO)), texto, re.M):
            continue
        exp_ok = True
        for run_dir in glob.glob(os.path.join(exp_dir, "*")):
            if not os.path.isdir(run_dir) or not \
               os.path.isfile(os.path.join(run_dir, "meta.yaml")):
                continue
            nombre = None
            tagfile = os.path.join(run_dir, "tags", "mlflow.runName")
            if os.path.isfile(tagfile):
                with open(tagfile, encoding="utf-8", errors="replace") as f:
                    nombre = f.read().strip()
            if not nombre:
                with open(os.path.join(run_dir, "meta.yaml"),
                          encoding="utf-8", errors="replace") as f:
                    m = re.search(r"run_name:\s*(\S+)", f.read())
                nombre = m.group(1) if m else os.path.basename(run_dir)
            params, metrics = {}, {}
            for p in glob.glob(os.path.join(run_dir, "params", "*")):
                with open(p, encoding="utf-8", errors="replace") as f:
                    params[os.path.basename(p)] = f.read().strip()
            for mfile in glob.glob(os.path.join(run_dir, "metrics", "*")):
                with open(mfile, encoding="utf-8", errors="replace") as f:
                    lineas = [l.split() for l in f.read().splitlines() if l.strip()]
                if lineas and len(lineas[-1]) >= 2:
                    try:
                        metrics[os.path.basename(mfile)] = float(lineas[-1][1])
                    except ValueError:
                        pass
            previo = encontrados.get(nombre)
            if previo is None or len(metrics) > len(previo["metrics"]):
                encontrados[nombre] = {"params": params, "metrics": metrics}
    dir_reg = os.path.join(base, "models", MODELO_REGISTRADO)
    registrado = os.path.isdir(dir_reg) and any(
        os.path.isdir(d) for d in glob.glob(os.path.join(dir_reg, "version-*")))
    alias_ok = os.path.isfile(os.path.join(dir_reg, "aliases", "champion"))
    return exp_ok, encontrados, registrado, alias_ok


def validar_mlflow(raiz, ref, hallazgos):
    seccion("P6 MLflow (S07)")
    pts = 0.0
    db = os.path.join(raiz, "mlflow.db")
    exp_ok, runs, registrado, alias_ok = False, {}, False, False
    if os.path.isfile(db):
        print("[INFO] leyendo metadatos de mlflow.db (backend sqlite por defecto)")
        exp_ok, runs, registrado, alias_ok = leer_runs_sqlite(db)
    if not exp_ok and not runs:
        print("[INFO] leyendo file store mlruns/ (MLflow antiguo o mlflow.db vacío)")
        exp_ok, runs, registrado, alias_ok = leer_runs_filestore(raiz)
    if check(exp_ok, "experimento '{}' presente en los registros de MLflow".format(EXPERIMENTO)):
        pts += 0.5
    for nombre in RUNS_REQUERIDOS:
        run = runs.get(nombre)
        tiene = run is not None
        con_params = tiene and len(run["params"]) >= 1
        if check(tiene and con_params,
                 "run '{}' presente con params (params: {})".format(
                     nombre, sorted(run["params"]) if run else "-")):
            pts += 1.0 / 3.0
    comparaciones, aciertos = 0, 0
    for nombre in RUNS_REQUERIDOS:
        run = runs.get(nombre, {"metrics": {}})
        for metrica in METRICAS:
            comparaciones += 1
            v = run["metrics"].get(metrica)
            vref = ref["modelos"][nombre][metrica]
            if check(cerca(v, vref, TOL_MODELOS),
                     "{}::{} = {} (ref {}, tol {})".format(nombre, metrica, v, vref, TOL_MODELOS)):
                aciertos += 1
    pts += 1.5 * aciertos / comparaciones
    puntuar("P6 MLflow", pts, 3.0)

    # Coherencia hallazgos <-> mlruns (alerta, sin puntaje).
    if runs and hallazgos:
        con_rmse = {n: r["metrics"].get("rmse_test") for n, r in runs.items()
                    if r["metrics"].get("rmse_test") is not None}
        if con_rmse:
            mejor_en_mlruns = min(con_rmse, key=con_rmse.get)
            if hallazgos.get("mejor_modelo") and \
               hallazgos["mejor_modelo"] != mejor_en_mlruns:
                alerta("hallazgos.mejor_modelo='{}' pero en mlruns el menor rmse_test es '{}'".format(
                    hallazgos["mejor_modelo"], mejor_en_mlruns))

    seccion("Bonus Model Registry (S07)")
    bonus = 0.0
    if check(registrado, "modelo '{}' registrado en el Model Registry".format(MODELO_REGISTRADO)):
        bonus += 0.5
    if check(alias_ok, "alias 'champion' asignado"):
        bonus += 0.5
    puntuar("Bonus Registry", bonus, 1.0)


# --------------------------------------------------------------------------- #
# P7 — App Streamlit (S05) · 2 pt
# --------------------------------------------------------------------------- #
def validar_app(raiz):
    seccion("P7 App Streamlit (S05)")
    pts = 0.0
    ruta_app = os.path.join(raiz, "app.py")
    codigo = None
    if os.path.isfile(ruta_app):
        with open(ruta_app, encoding="utf-8", errors="replace") as f:
            codigo = f.read()
    if codigo is None:
        check(False, "existe app.py")
        puntuar("P7 App", 0.0, 2.0)
        return
    try:
        arbol = ast.parse(codigo)
        compila = True
    except SyntaxError as e:
        compila = False
        arbol = None
        print("[FALLA] app.py no compila: {}".format(e))
    check(compila, "app.py compila (ast.parse)")
    atributos, cadenas = set(), []
    if arbol is not None:
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.Attribute):
                atributos.add(nodo.attr)
            if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
                cadenas.append(nodo.value)
    requisitos = [
        ("title", "st.title(...)"),
        ("sidebar", "st.sidebar (filtros en barra lateral)"),
        ("selectbox", "selectbox"),
        ("slider", "slider"),
        ("metric", "st.metric (KPI)"),
        ("pyplot", "st.pyplot(fig)"),
        ("cache_data", "@st.cache_data en la carga"),
    ]
    for attr, desc in requisitos:
        if check(attr in atributos, "AST contiene {}".format(desc)):
            pts += 0.2
    lee_data = any(("data/" in c) or ("data\\" in c) or c == "data" for c in cadenas)
    if check(lee_data, "lee un CSV de data/"):
        pts += 0.2
    ruta_req = os.path.join(raiz, "requirements.txt")
    con_streamlit = False
    if os.path.isfile(ruta_req):
        with open(ruta_req, encoding="utf-8", errors="replace") as f:
            con_streamlit = "streamlit" in f.read().lower()
    if check(con_streamlit, "requirements.txt existe y declara streamlit"):
        pts += 0.4
    puntuar("P7 App", pts, 2.0)


# --------------------------------------------------------------------------- #
# P8 — Teoría · 4 pt
# --------------------------------------------------------------------------- #
def validar_teoria(raiz):
    seccion("P8 Teoría (S01-S07)")
    ruta = os.path.join(raiz, "resultados", "respuestas_teoria.json")
    est = {}
    if os.path.isfile(ruta):
        try:
            with open(ruta, encoding="utf-8-sig") as f:
                est = json.load(f)
            if not isinstance(est, dict):
                est = {}
        except (json.JSONDecodeError, OSError, ValueError):
            est = {}
    check(bool(est), "resultados/respuestas_teoria.json existe y es JSON válido")
    pts = 0.0
    for p in sorted(RESPUESTAS):
        dada = str(est.get(p, "")).strip().lower()
        if check(dada == RESPUESTAS[p], "{}: '{}' (correcta: '{}')".format(p, dada or "-", RESPUESTAS[p])):
            pts += 0.5
    puntuar("P8 Teoría", pts, 4.0)


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def parsear_fecha(txt):
    if not txt:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, fmt)
        except ValueError:
            continue
    print("[INFO] fecha no reconocida (usa 'YYYY-MM-DD HH:MM'): {}".format(txt))
    return None


def main():
    parser = argparse.ArgumentParser(description="Corrector del simulacro parcial")
    parser.add_argument("entrega", nargs="?", default=".",
                        help="carpeta del proyecto o .zip de entrega")
    parser.add_argument("--inicio", default=None, help="inicio del examen 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--fin", default=None, help="fin del examen 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--correo", default=None, help="correo PUCP declarado del estudiante")
    args = parser.parse_args()

    print("=" * 68)
    print("SIMULACRO DE EVALUACIÓN PARCIAL — corrección automática (mpg)")
    print("Entrega     : {}".format(os.path.abspath(args.entrega)))
    print("Tolerancias : EDA {} · modelos {}".format(TOL, TOL_MODELOS))
    print("=" * 68)

    raiz, tmpdir = localizar_raiz(args.entrega)
    if raiz is None:
        print("[FALLA] no se encontró la carpeta del proyecto "
              "(se busca {} o .git).".format(NOTEBOOK))
        sys.exit(1)
    print("Raíz del proyecto: {}".format(raiz))

    print("\n-- Recomputando referencia desde 'mpg' --")
    ref = calcular_referencia()
    print("   filas limpias ref = {}, origen más eficiente = {}".format(
        ref["n_filas_limpio"], ref["origen_mas_eficiente"]))

    try:
        validar_entorno(raiz)
        validar_dvc(raiz, ref)
        hallazgos = validar_eda(raiz, ref)
        validar_git(raiz, parsear_fecha(args.inicio), parsear_fecha(args.fin),
                    args.correo)
        validar_mlflow(raiz, ref, hallazgos or {})
        validar_app(raiz)
        validar_teoria(raiz)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 68)
    print("RESUMEN DE PUNTAJE")
    base, bonus = 0.0, 0.0
    for item, obtenido, maximo in _puntajes:
        print("   {:<16} {:>5.2f} / {:.2f}".format(item, obtenido, maximo))
        if item.startswith("Bonus"):
            bonus += obtenido
        else:
            base += obtenido
    nota = min(20.0, base + bonus)
    print("-" * 68)
    print("NOTA (referencial, simulacro): {:.2f} / 20   (base {:.2f} + bonus {:.2f})".format(
        nota, base, bonus))
    if _alertas:
        print("\nALERTAS DE AUTORÍA (revisar manualmente):")
        for a in _alertas:
            print("   - {}".format(a))
    print("=" * 68)
    sys.exit(0)


if __name__ == "__main__":
    main()
