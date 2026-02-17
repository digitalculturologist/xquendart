import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
import math

# ─── Page Configuration ───
st.set_page_config(
    page_title="XquendArt",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State Initialization ───
if "mode" not in st.session_state:
    st.session_state.mode = "figura"
if "text_source" not in st.session_state:
    st.session_state.text_source = "direct"
if "parsed_lists" not in st.session_state:
    st.session_state.parsed_lists = {}
if "parsed_poems" not in st.session_state:
    st.session_state.parsed_poems = []
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None


# ═══════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════

def parse_txt_file(content):
    """
    Parses the poet's TXT file into word lists and poems.
    
    Format:
        === LISTA: Name ===
        indigenous_word | spanish_translation
        ...
        
        === POEMA ===
        poem text line 1
        poem text line 2
        ...
    """
    lists = {}
    poems = []
    
    lines = content.strip().split("\n")
    current_section = None
    current_name = None
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check for LISTA header
        if stripped.startswith("=== LISTA:") and stripped.endswith("==="):
            # Save previous section if any
            if current_section == "lista" and current_name:
                lists[current_name] = current_content
            elif current_section == "poema" and current_content:
                poems.append("\n".join(current_content))
            
            # Start new list
            current_section = "lista"
            current_name = stripped.replace("=== LISTA:", "").replace("===", "").strip()
            current_content = []
        
        # Check for POEMA header
        elif stripped == "=== POEMA ===":
            # Save previous section if any
            if current_section == "lista" and current_name:
                lists[current_name] = current_content
            elif current_section == "poema" and current_content:
                poems.append("\n".join(current_content))
            
            # Start new poem
            current_section = "poema"
            current_name = None
            current_content = []
        
        # Content lines
        elif stripped and current_section:
            if current_section == "lista":
                # Parse "indigenous_word | translation" format
                if "|" in stripped:
                    parts = stripped.split("|")
                    word = parts[0].strip()
                    translation = parts[1].strip() if len(parts) > 1 else ""
                    current_content.append({
                        "word": word,
                        "translation": translation
                    })
                else:
                    # No translation provided
                    current_content.append({
                        "word": stripped,
                        "translation": ""
                    })
            elif current_section == "poema":
                current_content.append(stripped)
    
    # Save last section
    if current_section == "lista" and current_name:
        lists[current_name] = current_content
    elif current_section == "poema" and current_content:
        poems.append("\n".join(current_content))
    
    return lists, poems


def get_default_font(size=20):
    """Returns a font object. Uses Noto Sans if available, else default."""
    font_paths = [
        "assets/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback to default font (scaled approximately)
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def remove_background(image):
    """Removes background from an image using rembg."""
    from rembg import remove
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    result_bytes = remove(img_bytes.read())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def image_to_grayscale_mask(image):
    """
    Converts an RGBA image (with background removed) to a grayscale mask.
    Transparent areas become white (255), subject retains grayscale values.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    
    # Extract alpha channel and grayscale
    r, g, b, a = image.split()
    grayscale = image.convert("L")  # Luminance
    
    # Where alpha is 0 (transparent), set to white (255 = empty)
    mask_array = np.array(grayscale, dtype=np.float64)
    alpha_array = np.array(a, dtype=np.float64)
    
    # Normalize alpha to 0-1
    alpha_norm = alpha_array / 255.0
    
    # Blend: where transparent, use white (255); where opaque, use grayscale
    result = mask_array * alpha_norm + 255.0 * (1.0 - alpha_norm)
    
    return Image.fromarray(result.astype(np.uint8), mode="L")


def generate_basic_shape(shape_name, width, height):
    """Generates a grayscale mask for basic geometric shapes."""
    img = Image.new("L", (width, height), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    margin = int(min(width, height) * 0.1)
    
    if shape_name == "Círculo":
        draw.ellipse(
            [margin, margin, width - margin, height - margin],
            fill=0
        )
    
    elif shape_name == "Corazón":
        # Heart shape using polygon approximation
        cx, cy = width // 2, height // 2
        scale = min(width, height) // 2 - margin
        points = []
        for i in range(360):
            t = math.radians(i)
            x = 16 * math.sin(t) ** 3
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            px = int(cx + x * scale / 17)
            py = int(cy + y * scale / 17)
            points.append((px, py))
        draw.polygon(points, fill=0)
    
    elif shape_name == "Estrella":
        cx, cy = width // 2, height // 2
        outer_r = min(width, height) // 2 - margin
        inner_r = outer_r * 0.4
        points = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = outer_r if i % 2 == 0 else inner_r
            points.append((
                int(cx + r * math.cos(angle)),
                int(cy + r * math.sin(angle))
            ))
        draw.polygon(points, fill=0)
    
    elif shape_name == "Triángulo":
        points = [
            (width // 2, margin),
            (margin, height - margin),
            (width - margin, height - margin)
        ]
        draw.polygon(points, fill=0)
    
    elif shape_name == "Cruz":
        arm_w = min(width, height) // 4
        cx, cy = width // 2, height // 2
        half = min(width, height) // 2 - margin
        # Vertical bar
        draw.rectangle(
            [cx - arm_w//2, cy - half, cx + arm_w//2, cy + half],
            fill=0
        )
        # Horizontal bar
        draw.rectangle(
            [cx - half, cy - arm_w//2, cx + half, cy + arm_w//2],
            fill=0
        )
    
    elif shape_name == "Óvalo":
        draw.ellipse(
            [margin, margin + height//6, width - margin, height - margin - height//6],
            fill=0
        )
    
    elif shape_name == "Diamante":
        cx, cy = width // 2, height // 2
        half_w = width // 2 - margin
        half_h = height // 2 - margin
        points = [
            (cx, cy - half_h),
            (cx + half_w, cy),
            (cx, cy + half_h),
            (cx - half_w, cy)
        ]
        draw.polygon(points, fill=0)
    
    elif shape_name == "Luna creciente":
        # Full circle
        draw.ellipse(
            [margin, margin, width - margin, height - margin],
            fill=0
        )
        # Subtract offset circle (white)
        offset = int(min(width, height) * 0.25)
        draw.ellipse(
            [margin + offset, margin, width - margin + offset, height - margin],
            fill=255
        )
    
    return img


def get_text_for_rendering(parsed_lists, parsed_poems, text_source, 
                           direct_text="", selected_list=None, selected_poem_idx=None):
    """
    Returns the final text string to be rendered, based on the text source.
    """
    if text_source == "direct":
        return direct_text
    elif text_source == "file_poem" and parsed_poems:
        idx = selected_poem_idx if selected_poem_idx is not None else 0
        if idx < len(parsed_poems):
            return parsed_poems[idx]
        return ""
    elif text_source == "file_list" and parsed_lists and selected_list:
        if selected_list in parsed_lists:
            words = [item["word"] for item in parsed_lists[selected_list]]
            return " ".join(words)
        return ""
    return direct_text


def fill_shape_with_text(mask, text, config):
    """
    Core rendering engine: fills a grayscale mask with text.
    
    mask: PIL Image in mode "L" (grayscale)
    text: string of words to render
    config: dict with all rendering parameters
    
    Returns: PIL Image (RGBA) and SVG string
    """
    mode = config.get("render_mode", "textura")
    density = config.get("density", 80)
    invert = config.get("invert", False)
    direction = config.get("direction", "lr")
    font_min = config.get("font_min", 10)
    font_max = config.get("font_max", 28)
    spacing = config.get("spacing", 4)
    text_color = config.get("text_color", "#000000")
    bg_color = config.get("bg_color", "#FFFFFF")
    bg_transparent = config.get("bg_transparent", False)
    loop_text = config.get("loop_text", True)
    output_width = config.get("output_width", 2000)
    output_height = config.get("output_height", 2000)
    
    # Resize mask to grid resolution
    mask_resized = mask.resize((density, density), Image.Resampling.LANCZOS)
    mask_array = np.array(mask_resized)
    
    if invert:
        mask_array = 255 - mask_array
    
    # Prepare output image
    if bg_transparent:
        output = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    else:
        # Parse hex color
        bg_r = int(bg_color[1:3], 16)
        bg_g = int(bg_color[3:5], 16)
        bg_b = int(bg_color[5:7], 16)
        output = Image.new("RGBA", (output_width, output_height), (bg_r, bg_g, bg_b, 255))
    
    draw = ImageDraw.Draw(output)
    
    # Parse text color
    tc_r = int(text_color[1:3], 16)
    tc_g = int(text_color[3:5], 16)
    tc_b = int(text_color[5:7], 16)
    
    # Prepare word list
    words = text.split()
    if not words:
        return output, ""
    
    word_index = 0
    svg_elements = []
    
    # Cell dimensions
    cell_w = output_width / density
    cell_h = output_height / density
    
    # Determine iteration order based on direction
    if direction == "lr":
        row_range = range(density)
        col_range_fn = lambda: range(density)
    elif direction == "rl":
        row_range = range(density)
        col_range_fn = lambda: range(density - 1, -1, -1)
    elif direction == "tb":
        # Top to bottom: iterate columns first, then rows
        row_range = range(density)
        col_range_fn = lambda: range(density)
    elif direction == "bt":
        row_range = range(density - 1, -1, -1)
        col_range_fn = lambda: range(density)
    else:
        row_range = range(density)
        col_range_fn = lambda: range(density)
    
    for row in row_range:
        for col in col_range_fn():
            pixel_val = mask_array[row, col]
            
            # Decide whether to place text based on render mode
            place_text = False
            font_scale = 1.0
            
            if mode == "textura":
                # Darker = more likely to place text, and larger
                if pixel_val < 200:  # Not white/near-white
                    place_text = True
                    # Scale: 0 (black) -> 1.0 (full size), 200 (light) -> 0.3 (small)
                    font_scale = 1.0 - (pixel_val / 255.0) * 0.7
            
            elif mode == "silueta":
                # Any non-white pixel gets text, uniform size
                if pixel_val < 240:
                    place_text = True
                    font_scale = 0.8
            
            elif mode == "contorno":
                # Edge detection: place text only near edges
                # Check if this pixel is near a boundary (dark next to light)
                if pixel_val < 240:
                    is_edge = False
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < density and 0 <= nc < density:
                            if mask_array[nr, nc] >= 240:
                                is_edge = True
                                break
                        else:
                            is_edge = True
                            break
                    if is_edge:
                        place_text = True
                        font_scale = 0.7
            
            if place_text:
                if not loop_text and word_index >= len(words):
                    break
                
                word = words[word_index % len(words)]
                word_index += 1
                
                # Calculate font size
                font_size = int(font_min + (font_max - font_min) * font_scale)
                font_size = max(font_min, min(font_max, font_size))
                
                font = get_default_font(font_size)
                
                # Position: center of cell
                x = int(col * cell_w + spacing)
                y = int(row * cell_h + spacing)
                
                # Draw text
                draw.text(
                    (x, y), 
                    word, 
                    fill=(tc_r, tc_g, tc_b, 255), 
                    font=font
                )
                
                # SVG element
                svg_elements.append(
                    f'<text x="{x}" y="{y + font_size}" '
                    f'font-size="{font_size}" '
                    f'fill="rgb({tc_r},{tc_g},{tc_b})" '
                    f'font-family="Noto Sans, sans-serif">'
                    f'{word}</text>'
                )
        
        if not loop_text and word_index >= len(words):
            break
    
    # Build SVG string
    svg_string = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{output_width}" height="{output_height}" '
        f'viewBox="0 0 {output_width} {output_height}">\n'
    )
    if not bg_transparent:
        svg_string += (
            f'  <rect width="{output_width}" height="{output_height}" '
            f'fill="{bg_color}"/>\n'
        )
    for elem in svg_elements:
        svg_string += f"  {elem}\n"
    svg_string += "</svg>"
    
    return output, svg_string


# ═══════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════

with st.sidebar:
    st.title("🌸 XquendArt")
    st.caption("Generador de caligramas para poetas indígenas")
    
    st.divider()
    
    # ── Mode Selection ──
    st.subheader("Modo")
    mode = st.radio(
        "Selecciona el modo:",
        ["Modo Figura", "Modo Lienzo"],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.mode = "figura" if mode == "Modo Figura" else "lienzo"
    
    st.divider()
    
    # ── Text Source ──
    st.subheader("Fuente de Texto")
    text_source = st.radio(
        "¿De dónde viene el texto?",
        ["Escribir directamente", "Subir archivo TXT", "Ordenar con Gemini"],
        index=0,
        label_visibility="collapsed"
    )
    
    direct_text = ""
    selected_list_name = None
    selected_poem_idx = None
    
    if text_source == "Escribir directamente":
        st.session_state.text_source = "direct"
        direct_text = st.text_area(
            "Escribe o pega tu texto:",
            height=150,
            placeholder="Escribe aquí las palabras o el poema..."
        )
    
    elif text_source == "Subir archivo TXT":
        uploaded_txt = st.file_uploader(
            "Sube tu archivo TXT:",
            type=["txt"],
            help="Formato: === LISTA: Nombre === y === POEMA ==="
        )
        if uploaded_txt is not None:
            content = uploaded_txt.read().decode("utf-8")
            parsed_lists, parsed_poems = parse_txt_file(content)
            st.session_state.parsed_lists = parsed_lists
            st.session_state.parsed_poems = parsed_poems
            
            if parsed_lists:
                st.success(f"✅ {len(parsed_lists)} lista(s) encontrada(s)")
            if parsed_poems:
                st.success(f"✅ {len(parsed_poems)} poema(s) encontrado(s)")
            
            # Let poet choose which to use
            source_options = []
            if parsed_poems:
                for i, poem in enumerate(parsed_poems):
                    preview = poem[:50] + "..." if len(poem) > 50 else poem
                    source_options.append(f"Poema {i+1}: {preview}")
            if parsed_lists:
                for name in parsed_lists:
                    source_options.append(f"Lista: {name}")
            
            if source_options:
                chosen = st.selectbox("Usar:", source_options)
                if chosen.startswith("Poema"):
                    idx = int(chosen.split(":")[0].replace("Poema ", "")) - 1
                    st.session_state.text_source = "file_poem"
                    selected_poem_idx = idx
                elif chosen.startswith("Lista:"):
                    name = chosen.replace("Lista: ", "")
                    st.session_state.text_source = "file_list"
                    selected_list_name = name
    
    elif text_source == "Ordenar con Gemini":
        st.session_state.text_source = "gemini"
        api_key = st.text_input(
            "Clave API de Gemini:",
            type="password",
            help="Tu clave personal. No se almacena."
        )
        st.markdown(
            "[¿Cómo obtener tu clave gratuita?]"
            "(https://aistudio.google.com/apikey)",
            unsafe_allow_html=True
        )
        if not api_key:
            st.warning("⚠️ Ingresa tu clave API para usar Gemini.")
        
        arrangement_style = st.selectbox(
            "Estilo de ordenamiento:",
            ["Flujo natural", "Contraste", "Repetición poética", "Aleatorio"]
        )
    
    st.divider()
    
    # ── Font Settings ──
    st.subheader("Tipografía")
    uploaded_font = st.file_uploader(
        "Subir fuente personalizada (.ttf):",
        type=["ttf"],
        help="Opcional. Se usa Noto Sans por defecto."
    )
    
    st.divider()
    
    # ── Output Settings ──
    st.subheader("Exportar")
    
    resolution_preset = st.selectbox(
        "Resolución:",
        ["Pequeña (1000px)", "Mediana (2000px)", "Grande (3000px)", "HD (4000px)", "Personalizada"]
    )
    
    if resolution_preset == "Personalizada":
        col1, col2 = st.columns(2)
        with col1:
            output_width = st.number_input("Ancho (px):", min_value=500, max_value=8000, value=2000)
        with col2:
            output_height = st.number_input("Alto (px):", min_value=500, max_value=8000, value=2000)
    else:
        size_map = {
            "Pequeña (1000px)": 1000,
            "Mediana (2000px)": 2000,
            "Grande (3000px)": 3000,
            "HD (4000px)": 4000
        }
        output_width = size_map[resolution_preset]
        output_height = size_map[resolution_preset]


# ═══════════════════════════════════════════
# MAIN AREA — MODO FIGURA
# ═══════════════════════════════════════════

if st.session_state.mode == "figura":
    st.header("🖼️ Modo Figura")
    st.markdown("Sube una imagen o elige una forma, y el texto llenará la figura.")
    
    # ── Image Source ──
    col_input, col_preview = st.columns([1, 1])
    
    with col_input:
        st.subheader("Imagen de referencia")
        
        image_source = st.radio(
            "Fuente de imagen:",
            ["Subir imagen", "Forma básica"],
            horizontal=True
        )
        
        mask = None
        
        if image_source == "Subir imagen":
            uploaded_img = st.file_uploader(
                "Sube tu imagen:",
                type=["png", "jpg", "jpeg", "webp"],
                help="Funciona mejor con fotos donde el sujeto es claro."
            )
            
            if uploaded_img is not None:
                original_image = Image.open(uploaded_img)
                st.image(original_image, caption="Imagen original", use_container_width=True)
                
                with st.spinner("🔄 Eliminando fondo..."):
                    try:
                        no_bg = remove_background(original_image)
                        mask = image_to_grayscale_mask(no_bg)
                        st.image(mask, caption="Máscara generada", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error al procesar imagen: {str(e)}")
        
        elif image_source == "Forma básica":
            shape_name = st.selectbox(
                "Elige una forma:",
                ["Círculo", "Corazón", "Estrella", "Triángulo", 
                 "Cruz", "Óvalo", "Diamante", "Luna creciente"]
            )
            mask = generate_basic_shape(shape_name, output_width, output_height)
            st.image(mask, caption=f"Forma: {shape_name}", use_container_width=True)
    
    # ── Rendering Controls ──
    with col_preview:
        st.subheader("Controles de renderizado")
        
        render_mode = st.selectbox(
            "Modo de relleno:",
            ["Textura", "Silueta", "Contorno"]
        )
        
        density_preset = st.select_slider(
            "Densidad:",
            options=["Baja", "Media", "Alta", "Ultra", "Manual"],
            value="Media"
        )
        
        density_map = {"Baja": 40, "Media": 80, "Alta": 150, "Ultra": 250}
        if density_preset == "Manual":
            density = st.slider("Densidad manual:", 20, 300, 80)
        else:
            density = density_map[density_preset]
        
        invert = st.toggle("Invertir claro/oscuro", value=False)
        
        direction = st.selectbox(
            "Dirección de lectura:",
            ["Izquierda → Derecha", "Derecha → Izquierda", 
             "Arriba → Abajo", "Abajo → Arriba"]
        )
        direction_map = {
            "Izquierda → Derecha": "lr",
            "Derecha → Izquierda": "rl",
            "Arriba → Abajo": "tb",
            "Abajo → Arriba": "bt"
        }
        
        col_font1, col_font2 = st.columns(2)
        with col_font1:
            font_min = st.number_input("Tamaño mín:", min_value=4, max_value=100, value=8)
        with col_font2:
            font_max = st.number_input("Tamaño máx:", min_value=4, max_value=100, value=24)
        
        spacing = st.slider("Espaciado entre palabras:", 0, 30, 4)
        
        loop_text = st.toggle("Repetir texto al agotarse", value=True)
        
        text_color = st.color_picker("Color del texto:", "#000000")
        
        bg_option = st.radio(
            "Fondo:",
            ["Color sólido", "Transparente"],
            horizontal=True
        )
        bg_transparent = bg_option == "Transparente"
        bg_color = "#FFFFFF"
        if not bg_transparent:
            bg_color = st.color_picker("Color de fondo:", "#FFFFFF")
    
    # ── Generate ──
    st.divider()
    
    # Get the text to render
    render_text = get_text_for_rendering(
        st.session_state.parsed_lists,
        st.session_state.parsed_poems,
        st.session_state.text_source,
        direct_text=direct_text,
        selected_list=selected_list_name,
        selected_poem_idx=selected_poem_idx
    )
    
    if mask is not None and render_text.strip():
        if st.button("✨ Generar caligrama", type="primary", use_container_width=True):
            config = {
                "render_mode": render_mode.lower(),
                "density": density,
                "invert": invert,
                "direction": direction_map[direction],
                "font_min": font_min,
                "font_max": font_max,
                "spacing": spacing,
                "text_color": text_color,
                "bg_color": bg_color,
                "bg_transparent": bg_transparent,
                "loop_text": loop_text,
                "output_width": output_width,
                "output_height": output_height,
            }
            
            with st.spinner("🎨 Generando caligrama..."):
                result_image, svg_string = fill_shape_with_text(mask, render_text, config)
                st.session_state.generated_image = result_image
                st.session_state.generated_svg = svg_string
            
            st.success("✅ ¡Caligrama generado!")
            st.image(result_image, caption="Tu caligrama", use_container_width=True)
            
            # Download buttons
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                buf = io.BytesIO()
                result_image.save(buf, format="PNG")
                st.download_button(
                    "📥 Descargar PNG",
                    data=buf.getvalue(),
                    file_name="xquendart_caligrama.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_dl2:
                st.download_button(
                    "📥 Descargar SVG",
                    data=svg_string,
                    file_name="xquendart_caligrama.svg",
                    mime="image/svg+xml",
                    use_container_width=True
                )
    
    elif mask is None:
        st.info("👆 Sube una imagen o elige una forma básica para comenzar.")
    elif not render_text.strip():
        st.info("👈 Escribe o carga texto en la barra lateral para comenzar.")


# ═══════════════════════════════════════════
# MAIN AREA — MODO LIENZO
# ═══════════════════════════════════════════

elif st.session_state.mode == "lienzo":
    st.header("🎨 Modo Lienzo")
    st.markdown("Dibuja libremente y tus trazos se convertirán en palabras.")
    
    st.info(
        "🚧 **Modo Lienzo en desarrollo.** "
        "Esta función estará disponible en la próxima actualización. "
        "Por ahora, usa el Modo Figura para generar caligramas."
    )
    
    # Placeholder for Phase 3
    st.markdown(
        """
        ### Próximamente:
        - 🖌️ Lienzo interactivo para dibujar con palabras en tiempo real
        - 🔄 Cambio de listas de palabras mientras dibujas
        - 📏 Tamaño de fuente ajustable (manual o por velocidad del cursor)
        - ↩️ Deshacer trazos
        - 📥 Exportar como PNG y SVG
        """
    )


# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════

st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "XquendArt — Creado con 💜 para los poetas indígenas de México<br>"
    "Xquenda_Lab · 2025"
    "</div>",
    unsafe_allow_html=True
)
