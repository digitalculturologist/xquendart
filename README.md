

# 🌸 XquendArt

**Generador de caligramas para poetas indígenas de México**

---

XquendArt es una aplicación web construida con [Streamlit](https://streamlit.io/) que permite crear **caligramas** — composiciones visuales donde el texto adopta la forma de una figura. Está diseñada especialmente para poetas que escriben en **lenguas indígenas mexicanas** (náhuatl, maya, zapoteco, mixteco, purépecha, etc.), respetando los caracteres especiales propios de estas lenguas: oclusivas glotales (ʼ), apóstrofos internos (k'iin, ts'o'ok), guiones en palabras compuestas (ni-k-tlazohtla), dos puntos para vocales largas (tuka:ri) y toda la variedad de caracteres Unicode con diacríticos.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green)

---

## 📋 Tabla de contenidos

- [Características](#-características)
- [Capturas de pantalla](#-capturas-de-pantalla)
- [Requisitos previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Uso rápido](#-uso-rápido)
- [Guía completa](#-guía-completa)
  - [Fuentes de texto](#1-fuentes-de-texto)
  - [Modo Figura](#2-modo-figura)
  - [Modos de relleno](#3-modos-de-relleno)
  - [Controles de renderizado](#4-controles-de-renderizado)
  - [Fondo](#5-opciones-de-fondo)
  - [Título y firma](#6-título-y-firma-del-autor)
  - [Tipografía personalizada](#7-tipografía-personalizada)
  - [Exportación](#8-exportación)
- [Formato del archivo TXT](#-formato-del-archivo-txt)
- [Integración con IA (Gemini / Gemma)](#-integración-con-ia-gemini--gemma)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Solución de problemas](#-solución-de-problemas)
- [Hoja de ruta](#-hoja-de-ruta)
- [Créditos](#-créditos)

---

## ✨ Características

| Característica | Descripción |
|---|---|
| **3 modos de relleno** | *Textura* (variación tonal con cuadrícula de ocupación), *Silueta* (relleno sólido por escaneo) y *Contorno* (solo los bordes de la figura) |
| **Imagen o forma** | Sube cualquier fotografía (el fondo se elimina automáticamente) o elige entre 8 formas geométricas básicas |
| **Texto directo, archivo o IA** | Escribe tu texto, carga un archivo `.txt` con listas y poemas, o deja que un modelo de IA ordene tus palabras |
| **Integración con Gemini / Gemma** | Usa la API de Google AI Studio para ordenar listas de palabras indígenas en poemas o prosa, con selección de modelo y control de creatividad |
| **Respeto a lenguas indígenas** | Preserva oclusivas glotales, apóstrofos internos, guiones compuestos, vocales largas con dos puntos y todos los caracteres Unicode con diacríticos |
| **4 direcciones de lectura** | Izquierda→Derecha, Derecha→Izquierda, Arriba→Abajo, Abajo→Arriba |
| **Fondos flexibles** | Color sólido, transparente o imagen de fondo con 5 modos de ajuste (rellenar, ajustar, estirar, original, mosaico) |
| **Título y firma** | Superpone título y nombre del autor con posición, tamaño y color independientes |
| **Tipografía personalizada** | Sube tu propia fuente `.ttf` o usa Noto Sans (incluida) |
| **Exportación dual** | Descarga tu caligrama como **PNG** (imagen raster) o **SVG** (vectorial, escalable sin pérdida) |
| **Múltiples resoluciones** | Desde 1000px hasta 4000px, o dimensiones personalizadas hasta 8000px |
| **Soporte HEIC/HEIF** | Acepta fotos directamente desde iPhone/iPad sin conversión previa |

---

## 📸 Capturas de pantalla

> *Agrega aquí capturas de pantalla de tu aplicación mostrando los diferentes modos y resultados.*

---

## 📦 Requisitos previos

- **Python 3.9** o superior
- **pip** (gestor de paquetes de Python)
- (Opcional) Una **clave API de Google AI Studio** si deseas usar la función de ordenamiento con IA

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/xquendart.git
cd xquendart
```

### 2. Crear un entorno virtual (recomendado)

```bash
python -m venv venv

# En Linux/macOS:
source venv/bin/activate

# En Windows:
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si no tienes un archivo `requirements.txt`, instala las dependencias manualmente:

```bash
pip install streamlit numpy pillow scipy rembg google-generativeai pillow-heif
```

> **Nota sobre `rembg`:** Este paquete descarga un modelo de segmentación (~170 MB) la primera vez que se usa. Requiere conexión a internet en la primera ejecución.

### 4. (Opcional) Colocar la fuente tipográfica

Para obtener la mejor compatibilidad con lenguas indígenas, coloca el archivo `NotoSans-Regular.ttf` en la carpeta `assets/`:

```
xquendart/
├── assets/
│   └── NotoSans-Regular.ttf
├── app.py
└── requirements.txt
```

Puedes descargar Noto Sans desde [Google Fonts](https://fonts.google.com/noto/specimen/Noto+Sans).

### 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## ⚡ Uso rápido

1. **Abre la aplicación** en tu navegador.
2. **Escribe tu texto** en la barra lateral (o carga un archivo / usa la IA).
3. **Sube una imagen** o elige una forma básica en el panel principal.
4. **Ajusta los controles** (modo de relleno, densidad, colores, etc.).
5. **Haz clic en "✨ Generar caligrama"**.
6. **Descarga** tu caligrama en PNG o SVG.

---

## 📖 Guía completa

### 1. Fuentes de texto

En la barra lateral, elige de dónde viene el texto para tu caligrama:

#### a) Escribir directamente
Escribe o pega tu texto en el área de texto. Puede ser un poema, una lista de palabras, una frase, etc.

#### b) Subir archivo TXT
Carga un archivo `.txt` con formato especial (ver [Formato del archivo TXT](#-formato-del-archivo-txt)). Puedes incluir múltiples listas de palabras y poemas en un solo archivo.

#### c) Ordenar con IA
Proporciona una lista de palabras en tu lengua indígena con sus traducciones, y un modelo de IA las ordenará en un poema o prosa poética. Ver [Integración con IA](#-integración-con-ia-gemini--gemma) para más detalles.

---

### 2. Modo Figura

Es el modo principal de la aplicación. El texto adopta la forma de una imagen o figura geométrica.

#### Subir imagen
- Formatos aceptados: PNG, JPG, JPEG, WebP, HEIC, HEIF.
- La aplicación **elimina automáticamente el fondo** usando inteligencia artificial (`rembg`).
- La máscara resultante se almacena en caché: si cambias otros parámetros sin cambiar la imagen, no se vuelve a procesar.
- Las imágenes grandes se redimensionan automáticamente a un máximo de 2048px para optimizar el procesamiento.

#### Formas básicas
Si no tienes una imagen, elige entre 8 formas geométricas prediseñadas:

| Forma | Descripción |
|---|---|
| Círculo | Forma circular centrada |
| Corazón | Curva cardioide paramétrica |
| Estrella | Estrella de 5 puntas |
| Triángulo | Triángulo equilátero |
| Cruz | Cruz simétrica |
| Óvalo | Elipse horizontal |
| Diamante | Rombo centrado |
| Luna creciente | Luna con superposición de elipses |

---

### 3. Modos de relleno

| Modo | Algoritmo | Mejor para |
|---|---|---|
| **Textura** | Cuadrícula de ocupación. El tamaño de cada palabra varía según la luminosidad del píxel correspondiente: zonas oscuras = texto más grande, zonas claras = texto más pequeño. Incluye un segundo pase para rellenar huecos restantes. | Fotografías con detalle tonal, retratos, paisajes |
| **Silueta** | Escaneo por líneas (horizontal o vertical). Llena toda la silueta con texto justificado, distribuyendo las palabras uniformemente dentro de cada segmento. | Formas con bordes claros, siluetas planas |
| **Contorno** | Detección de bordes por erosión morfológica. Solo coloca texto a lo largo del perímetro de la figura, con dilatación controlada para que el borde sea legible. | Efectos minimalistas, contornos de figuras |

---

### 4. Controles de renderizado

| Control | Rango | Descripción |
|---|---|---|
| **Densidad** | Baja (40) / Media (80) / Alta (150) / Ultra (250) / Manual (20–300) | Controla qué tan densamente se llena la figura con texto. Mayor densidad = más palabras, más detalle, pero más tiempo de procesamiento. |
| **Invertir claro/oscuro** | Sí / No | Invierte la máscara: el texto llena lo que antes era fondo y viceversa. |
| **Dirección de lectura** | 4 opciones | Determina el orden en que se colocan las palabras: izquierda→derecha, derecha→izquierda, arriba→abajo, abajo→arriba. |
| **Tamaño mínimo** | 4–100 px | Tamaño mínimo de fuente para las palabras. |
| **Tamaño máximo** | 4–100 px | Tamaño máximo de fuente. En modo Textura, las zonas más oscuras usan tamaños cercanos al máximo. |
| **Espaciado** | 0–30 px | Espacio entre palabras (en píxeles). |
| **Repetir texto** | Sí / No | Si el texto se agota antes de llenar la figura, vuelve a empezar desde el principio. Desactivar esta opción puede dejar áreas vacías. |
| **Color del texto** | Selector de color | Color con el que se renderizan todas las palabras. |

> **💡 Consejo:** Para densidad "Ultra" (250), se recomienda usar resolución "Grande" (3000px) o "HD" (4000px) para obtener los mejores resultados.

---

### 5. Opciones de fondo

| Opción | Descripción |
|---|---|
| **Color sólido** | Elige cualquier color como fondo (blanco por defecto). |
| **Transparente** | El fondo es transparente (útil para composición en otros programas). Se exporta como PNG con canal alfa. |
| **Imagen de fondo** | Sube una imagen que se usará como fondo detrás del texto. |

#### Modos de ajuste de imagen de fondo

| Modo | Comportamiento |
|---|---|
| **Rellenar** | Escala la imagen para cubrir todo el lienzo. Puede recortar los bordes. |
| **Ajustar** | Escala la imagen para que sea completamente visible. Puede dejar márgenes transparentes. |
| **Estirar** | Deforma la imagen para llenar exactamente el lienzo. |
| **Original** | Coloca la imagen en su tamaño real, centrada. |
| **Mosaico** | Repite la imagen como un patrón de baldosas. |

---

### 6. Título y firma del autor

Dentro del desplegable **"📝 Título y firma del autor"**, puedes agregar:

- **Título del caligrama:** Se renderiza sobre el caligrama terminado. Acepta cualquier carácter, idioma o símbolo.
- **Firma / Nombre del autor:** Se renderiza en una esquina o borde. Puedes incluir tu nombre, seudónimo, arroba de redes sociales o sitio web.

Cada uno tiene controles independientes de:
- **Posición:** 6 opciones (arriba centrado, abajo centrado, y las 4 esquinas).
- **Tamaño de fuente:** Slider independiente.
- **Color:** Selector independiente.

> **Nota:** Si el título y la firma comparten el mismo borde (ambos arriba o ambos abajo), la aplicación los separa automáticamente para que no se superpongan.

---

### 7. Tipografía personalizada

En la sección **"Tipografía"** de la barra lateral:

1. Haz clic en **"Subir fuente personalizada (.ttf)"**.
2. Selecciona un archivo `.ttf` desde tu computadora.
3. La fuente se aplicará a todo el caligrama (texto, título y firma).
4. Para volver a la fuente predeterminada, haz clic en **"🔄 Restablecer a Noto Sans"**.

> **Recomendación:** Si trabajas con lenguas indígenas que usan caracteres especiales, asegúrate de que tu fuente personalizada los soporte. Noto Sans tiene excelente cobertura Unicode.

---

### 8. Exportación

#### Resoluciones disponibles

| Preset | Dimensiones | Uso recomendado |
|---|---|---|
| Pequeña | 1000 × 1000 px | Vista previa, redes sociales |
| Mediana | 2000 × 2000 px | Uso general, publicaciones |
| Grande | 3000 × 3000 px | Impresión de calidad |
| HD | 4000 × 4000 px | Impresión de alta calidad, posters |
| Personalizada | Hasta 8000 × 8000 px | Necesidades específicas |

#### Formatos de descarga

- **PNG:** Imagen raster de alta calidad. Soporta transparencia. Ideal para compartir en redes sociales o imprimir.
- **SVG:** Imagen vectorial. Cada palabra es un elemento `<text>` individual. Escalable sin pérdida de calidad. Ideal para edición posterior en Inkscape, Illustrator, etc.

> **Nota sobre SVG:** El archivo SVG usa la fuente "Noto Sans" por referencia. Si abres el SVG en otra computadora, asegúrate de tener la fuente instalada, o el navegador/editor usará una fuente de respaldo.

---

## 📄 Formato del archivo TXT

XquendArt usa un formato de texto plano sencillo para organizar listas de palabras y poemas. Puedes incluir múltiples listas y múltiples poemas en un solo archivo.

### Estructura

```
=== LISTA: Nombre de la lista ===
palabra1 | traducción1
palabra2 | traducción2
palabra3 | traducción3
palabra4

=== LISTA: Otra lista ===
palabra5 | traducción5
palabra6 | traducción6

=== POEMA: Título del poema ===
verso uno del poema
verso dos del poema
verso tres del poema

=== POEMA: Otro título de otro poema ===
primer verso del segundo poema
segundo verso
tercer verso
```

### Reglas

- **Listas:** Comienzan con `=== LISTA: NombreDeLaLista ===`. Cada línea contiene una palabra, opcionalmente seguida de `|` y su traducción al español.
- **Poemas:** Comienzan con `=== POEMA: TituloDelPoema ===`. Cada línea posterior es un verso del poema.
- Las líneas vacías dentro de una sección se ignoran.
- La traducción es opcional. Si no la incluyes, simplemente escribe la palabra sola.
- Puedes mezclar listas y poemas en cualquier orden.

### Ejemplo completo

```
=== LISTA: Naturaleza Náhuatl ===
xochitl | flor
atl | agua
tonatiuh | sol
metztli | luna
ehecatl | viento
tlalli | tierra
quiahuitl | lluvia
citlalli | estrella
ilhuicatl | cielo
cuauhtli | águila

=== LISTA: Emociones Maya ===
k'iin | sol / día
ha' | agua
ik' | viento / espíritu
kaan | cielo
lu'um | tierra
ja' | lluvia

=== POEMA: Xon Ahuiyacan ===
Ica xon ahuiyacan ihuinti xochitli, tomac mani, aya.
Ma on te ya aquiloto xochicozquitl.
In toquiappancaxochiuh, tla celia xochitli,cueponia xochitli.
```

---

## 🤖 Integración con IA (Gemini / Gemma)

La función **"Ordenar con IA"** usa la API de Google AI Studio para tomar una lista de palabras indígenas y ordenarlas como un poema o prosa poética.

### Configuración

1. Obtén una **clave API gratuita** en [Google AI Studio](https://aistudio.google.com/apikey).
2. Pega la clave en el campo "Clave API de Google AI Studio" en la barra lateral.
3. Selecciona un modelo, estilo y formato de salida.

### Modelos disponibles

| Modelo | Velocidad | Límite diario aprox. | Mejor para |
|---|---|---|---|
| `gemini-flash-latest` | ⚡ Rápido | ~20 generaciones/día | Uso general, buena calidad |
| `gemini-3-flash-preview` | ⚡⚡ Muy rápido | ~20 generaciones/día | Textos largos |
| `gemini-3pro-preview` | 🧠 Más inteligente | ~20 generaciones/día | Resultados de mayor calidad |
| `gemma-3-27b-it` | ⚡ Rápido | ~14,400 generaciones/día | Uso intensivo, sin límite práctico |

> **💡 Consejo:** Si necesitas hacer muchas pruebas, usa **Gemma** — tiene un límite diario mucho más generoso.

### Estilos de ordenamiento

| Estilo | Descripción |
|---|---|
| **Flujo natural** | Ordena de lo terrenal a lo celestial, de lo concreto a lo abstracto. |
| **Contraste** | Alterna conceptos opuestos: luz/oscuridad, tierra/cielo. |
| **Repetición poética** | Crea patrones rítmicos repitiendo palabras clave, como en la poesía oral. |
| **Aleatorio** | Mezcla las palabras al azar (no requiere API Key). |

### Control de creatividad (temperatura)

- **0.0:** Resultado predecible y repetitivo. Útil para reproduciblidad.
- **1.0:** Balance entre creatividad y coherencia (valor predeterminado).
- **2.0:** Resultado caótico y original. Puede inventar combinaciones inesperadas.

### Validación automática

La aplicación verifica que el modelo haya usado las palabras de tu lista y no haya inventado palabras nuevas. Si más del 70% del texto generado no coincide con tu lista, recibirás una advertencia.

---

## 📁 Estructura del proyecto

```
xquendart/
│
├── app.py                  # Código principal de la aplicación
├── requirements.txt        # Dependencias de Python
├── README.md               # Este archivo
│
├── assets/
│   └── NotoSans-Regular.ttf   # Fuente predeterminada (opcional)
│
└── examples/               # (Opcional) Archivos de ejemplo
    ├── ejemplo_nahuatl.txt
    └── ejemplo_maya.txt
```

---

## ❓ Solución de problemas

### La eliminación de fondo falla o es lenta

- **Primera ejecución:** `rembg` descarga un modelo de ~170 MB la primera vez. Asegúrate de tener conexión a internet.
- **Sin GPU:** En máquinas sin GPU, el procesamiento puede tardar 10–30 segundos por imagen. El resultado se almacena en caché para que no se repita.
- **Imágenes problemáticas:** Funciona mejor con fotos donde el sujeto se distingue claramente del fondo. Fotos con fondos muy complejos pueden dar resultados imperfectos.

### El caligrama tiene zonas vacías

- **Activa "Repetir texto al agotarse"** si tu texto es corto.
- **Aumenta la densidad** (prueba Alta o Ultra).
- **Reduce el tamaño mínimo de fuente** (hasta 4px).
- **Reduce el espaciado** entre palabras.

### Caracteres especiales no se muestran correctamente

- Usa la fuente **Noto Sans** (incluida) o sube una fuente `.ttf` con buena cobertura Unicode.
- Verifica que tu archivo de texto esté codificado en **UTF-8**.

### Error de API de Gemini

- Verifica que tu clave API sea válida y esté activa.
- Si recibes errores de cuota (quota), cambia a **Gemma** o espera al día siguiente.
- Si el modelo inventa palabras, **baja la temperatura** (creatividad) a 0.5–0.8.

### La aplicación es lenta

- **Resolución:** Reduce la resolución de salida. La diferencia entre 2000px y 4000px cuadruplica el número de píxeles a procesar.
- **Densidad:** "Ultra" (250) es significativamente más lenta que "Media" (80).
- **Modo Textura:** Es el modo más pesado computacionalmente debido al doble pase de relleno. "Silueta" y "Contorno" son más rápidos.

---

## 🗺️ Hoja de ruta

- [x] Modo Figura con 3 modos de relleno
- [x] Eliminación automática de fondo
- [x] 8 formas geométricas básicas
- [x] Integración con Gemini / Gemma
- [x] Título y firma del autor
- [x] Fondos con imagen y 5 modos de ajuste
- [x] Exportación PNG + SVG
- [x] Tipografía personalizada
- [ ] Crear secciones dentro de los caligramas para colocar texto personalizado en ellas
- [ ] **Modo Lienzo** — dibujo interactivo con palabras en tiempo real
- [ ] Paletas de colores múltiples (gradientes, multicolor por palabra)
- [ ] Rotación de texto (palabras en ángulo)
- [ ] Más formas geométricas
- [ ] Galería de caligramas de ejemplo

---

## 🙏 Créditos

- **XquendArt** — Desarrollado por @digitalculturologist en Xquenda_Lab, 2026.
- Diseñado para los poetas indígenas de México, libre de usarse por cualquiera.
- Construido con [Streamlit](https://streamlit.io/), [Pillow](https://python-pillow.org/), [rembg](https://github.com/danielgatis/rembg), [SciPy](https://scipy.org/) y [Google Generative AI](https://ai.google.dev/).

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
