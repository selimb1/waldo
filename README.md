# Auditoría Aérea de Activos – WALDO (MVP)

Este MVP procesa imágenes aéreas (dron) con un modelo YOLOv8 (por ejemplo WALDO) para detectar activos (vehículos, maquinaria, contenedores), concilia los resultados con un `inventory.csv` y genera un **informe HTML** con tablas, gráficos simples y un mapa de hallazgos.

> **Requisitos**: Python 3.10+, `pip install -r requirements.txt`. Para detección real necesitas un modelo YOLOv8 entrenado (por ej. WALDO). Puedes usar pesos `.pt` compatibles de YOLOv8.

## Estructura

```
waldo_audit_mvp/
├─ app/
│  ├─ main.py              # CLI: run all
│  ├─ detect.py            # detección YOLO (ultralytics)
│  ├─ reconcile.py         # conciliación con inventario
│  ├─ report.py            # informe HTML
│  ├─ utils.py             # utilidades
│  ├─ config.yaml          # configuración
├─ data/
│  ├─ inputs/              # coloca aquí tus imágenes (.jpg/.png)
│  ├─ outputs/             # resultados (detections.csv, report.html)
│  ├─ inventory.csv        # inventario contable (ejemplo)
│  └─ sites.csv            # catálogo de obras/sitios (ejemplo)
├─ models/
│  └─ waldo.pt             # (coloca aquí tus pesos YOLOv8/WALDO)
├─ templates/
│  └─ report.html.j2       # plantilla Jinja2 para informe
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

## Uso rápido

1) Instala dependencias:

```bash
pip install -r requirements.txt
```

2) Copia tus imágenes a `data/inputs/`. Coloca los pesos YOLO en `models/waldo.pt` (o ajusta `config.yaml`).

3) Ejecuta la tubería completa:

```bash
python -m app.main
```

4) El informe se guarda en `data/outputs/report.html` y también se exporta a PDF (`report.pdf`) y Excel (`report.xlsx`).

El pipeline ejecuta tres etapas principales:

1. **Detección y miniaturas** (`app/detect.py`): procesa cada imagen en `data/inputs/`. Con `demo_mode: true` genera detecciones sintéticas reproducibles, y si lo desactivas cargará el modelo YOLO configurado (por defecto `models/waldo.pt`) para producir resultados reales junto con recortes JPEG en `data/outputs/thumbnails/`. Además extrae metadatos GPS/EXIF (latitud, longitud y altitud cuando están disponibles) y los vuelca en `data/outputs/detections.csv` para georreferenciar cada hallazgo.
2. **Conciliación contable** (`app/reconcile.py`): compara las detecciones agregadas por obra y clase con el inventario (`data/inventory.csv`) y escribe `data/outputs/reconciliation.csv`.
3. **Informe técnico** (`app/report.py`): usa la plantilla Jinja `templates/report.html.j2` para renderizar un dashboard HTML con tablas, badges de estado, mapa (Folium) y galería de evidencias, y luego exporta el resultado a PDF y Excel.

Puedes ajustar rutas, clases a detectar y parámetros del modo demo editando `app/config.yaml`.

## Datos de ejemplo
Incluí un `inventory.csv` y `sites.csv` de muestra. El script puede funcionar con imágenes reales o, si no hay pesos, también puede correrse en **modo demo** (simula detecciones) activando `demo_mode: true` en `config.yaml`.

## Docker (CPU)
```bash
docker build -t waldo-audit .
docker run --rm -v $PWD/data:/app/data waldo-audit
```

> Para GPU deberás ajustar el `Dockerfile` a una imagen base con CUDA y lanzar con `--gpus all`.

## Nota legal
Asegúrate de contar con permisos de vuelo (ANAC u organismo local) y respetar privacidad/seguridad. Este MVP es sólo con fines educativos y debe validarse antes de uso profesional.
# waldo
