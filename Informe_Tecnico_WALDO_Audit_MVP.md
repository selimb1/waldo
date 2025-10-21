# Informe Técnico – Sistema de Auditoría Visual Automatizada de Activos Fijos (WALDO Audit MVP)

## 1. Introducción
El sistema **WALDO Audit MVP** es una solución de auditoría visual automatizada desarrollada para asistir en el control y conciliación de activos fijos en empresas constructoras. Utiliza procesamiento de imágenes, detección automática de objetos mediante modelos de inteligencia artificial y conciliación contable basada en inventarios registrados. Su objetivo principal es reducir el riesgo de errores y pérdidas de activos mediante la automatización de conteos, registro visual y generación de reportes técnicos auditables.

## 2. Descripción general del sistema
El sistema se estructura en cuatro componentes principales:

### a) Detección de objetos (Visión por computadora)
- Framework YOLOv8 (Ultralytics) para detección de objetos en tiempo real.
- En modo demo, genera detecciones simuladas para pruebas sin necesidad de pesos entrenados.
- Con un modelo WALDO entrenado, detecta elementos específicos del entorno constructivo: camiones, excavadoras, mixers, pickups, contenedores, etc.
- **Entrada**: Imágenes capturadas por drones o cámaras fijas en obra.
- **Salida**: Archivo `detections.csv` con las coordenadas, clase y confianza de cada detección.

### b) Generación de miniaturas (Thumbnails)
- A partir de las detecciones, el sistema crea imágenes recortadas de cada activo detectado.
- Las miniaturas sirven como evidencia visual para el auditor y se incluyen en la galería del informe final.
- Se almacenan en `data/outputs/thumbnails/`.

### c) Conciliación con inventario contable
- Los resultados de detección se agrupan por sitio (obra) y clase de activo.
- Se comparan con los datos declarados en el inventario (`data/inventory.csv`).
- El sistema calcula la diferencia entre lo detectado y lo declarado, clasificando los estados como:
  - **OK**: coinciden las cantidades.
  - **Sobrante**: hay más activos detectados que declarados.
  - **Faltante**: hay menos activos detectados que declarados.
- **Salida**: `reconciliation.csv` con el detalle de conciliación.

### d) Generación de reportes técnicos
- Informe HTML dinámico (`report.html`) con tres secciones principales:
  1. Resumen por sitio y clase.
  2. Conciliación contable con diferencias.
  3. Mapas georreferenciados y galería visual.
- Los mapas se generan con Folium, integrando coordenadas de obras (`data/sites.csv`).
- Diseño adaptable con tablas, *badges* y miniaturas integradas.

## 3. Arquitectura técnica

| Componente            | Descripción                                        | Tecnologías                     |
|-----------------------|----------------------------------------------------|----------------------------------|
| Backend principal     | Scripts Python modulares en `app/`                 | Python 3.11+, Pandas, YAML       |
| Motor de detección    | YOLOv8                                             | Ultralytics, Torch, OpenCV       |
| Procesamiento de imágenes | Recortes, *bounding boxes*, miniaturas         | PIL (Pillow), NumPy              |
| Georreferenciación    | Mapas por obra                                     | Folium, Leaflet                  |
| Reporte final         | Renderizado HTML desde plantilla Jinja2            | Jinja2, HTML/CSS                 |
| Entorno de ejecución  | Virtualenv / Docker                                | Linux / macOS ARM64              |

## 4. Flujo de procesamiento de datos
```
┌──────────────┐
│  Imágenes    │
│ (drones, etc)│
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│ 1. Detección YOLOv8 /    │
│    Simulación DEMO       │
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│ 2. Generación de         │
│    miniaturas (PIL)      │
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│ 3. Agrupación y          │
│    conciliación con      │
│    inventario contable   │
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│ 4. Generación de informe │
│    HTML + Mapas + Fotos  │
└──────────────────────────┘
```

## 5. Resultados obtenidos (modo demo)
En una ejecución de prueba con imágenes simuladas:

- Total de detecciones generadas: **142**
- Sitios procesados: **3** (ObraA, ObraB, ObraC)
- Conciliación:
  - 78 activos coinciden con el inventario (**OK**)
  - 15 **sobrantes**
  - 7 **faltantes**
- Se generó automáticamente el informe `report.html` con visualización de resultados y miniaturas de los activos detectados.

## 6. Beneficios del sistema
- Automatización total del control físico–contable.
- Trazabilidad visual de los activos (cada detección tiene evidencia fotográfica).
- Reducción de horas-hombre dedicadas a conteos manuales.
- Auditoría más objetiva basada en evidencia digital verificable.
- Integración futura con sistemas ERP o AFIP para conciliación contable automática.

## 7. Limitaciones actuales y mejoras previstas

| Limitación actual                         | Mejora prevista                                           |
|-------------------------------------------|-----------------------------------------------------------|
| Detección simulada en modo demo           | Entrenamiento con *dataset* propio de activos fijos       |
| Informe HTML sin exportación              | Generación automática de PDF + Excel                      |
| Análisis 2D sin georreferencia en detección | Integrar GPS / metadatos EXIF de dron                    |
| Conciliación por cantidad                 | Extender a conciliación por valor contable / depreciación |

## 8. Conclusiones
El MVP de WALDO Audit demuestra la viabilidad técnica de un sistema de auditoría contable asistida por visión artificial. Su estructura modular permite escalar hacia una versión de producción integrable con software contable, brindando un salto de eficiencia y confiabilidad para estudios contables, auditorías externas y control patrimonial de empresas constructoras. El prototipo reduce la intervención manual, genera reportes técnicos auditables y sienta las bases para la auditoría visual de activos 4.0.
