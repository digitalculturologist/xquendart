import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import json
import math
import tempfile
import html
import base64
import streamlit.components.v1 as components
from scipy import ndimage

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

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
if "mode_selector" not in st.session_state:
    st.session_state.mode_selector = "Modo Figura"
if "text_source" not in st.session_state:
    st.session_state.text_source = "direct"
if "direct_text_value" not in st.session_state:
    st.session_state.direct_text_value = ""
if "parsed_lists" not in st.session_state:
    st.session_state.parsed_lists = {}
if "parsed_poems" not in st.session_state:
    st.session_state.parsed_poems = []
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "generated_svg" not in st.session_state:
    st.session_state.generated_svg = None
if "custom_font_path" not in st.session_state:  # NEW
    st.session_state.custom_font_path = None
if "custom_font_bytes" not in st.session_state:
    st.session_state.custom_font_bytes = None
if "custom_font_name" not in st.session_state:
    st.session_state.custom_font_name = None
if "gemini_result" not in st.session_state:
    st.session_state.gemini_result = None
if "cached_mask" not in st.session_state:
    st.session_state.cached_mask = None
if "cached_mask_key" not in st.session_state:
    st.session_state.cached_mask_key = None
if "cached_original" not in st.session_state:
    st.session_state.cached_original = None
if "font_uploader_key" not in st.session_state:
    st.session_state.font_uploader_key = 0
if "query_mode_loaded" not in st.session_state:
    st.session_state.query_mode_loaded = False
if "query_text_loaded" not in st.session_state:
    st.session_state.query_text_loaded = False

if not st.session_state.query_mode_loaded:
    query_mode = st.query_params.get("mode")
    if query_mode in ("figura", "lienzo"):
        st.session_state.mode = query_mode
        st.session_state.mode_selector = "Modo Figura" if query_mode == "figura" else "Modo Lienzo"
    st.session_state.query_mode_loaded = True

if not st.session_state.query_text_loaded:
    query_text = st.query_params.get("text")
    if query_text:
        st.session_state.direct_text_value = query_text
    st.session_state.query_text_loaded = True

# ── Font Cache ──
_font_cache = {}
APP_DIR = os.path.dirname(__file__)

AI_MODEL_OPTIONS = [
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemma-4-31b-it",
]
AI_MODEL_LIMITS = {
    "gemini-flash-latest": {
        "description": "Rápido y balanceado",
        "rpm": "5 RPM",
        "tpm": "250K TPM",
        "rpd": "20 RPD",
    },
    "gemini-3.1-flash-lite": {
        "description": "Más margen diario para iterar",
        "rpm": "15 RPM",
        "tpm": "250K TPM",
        "rpd": "500 RPD",
    },
    "gemini-3.5-flash": {
        "description": "Flash más reciente para textos largos",
        "rpm": "5 RPM",
        "tpm": "250K TPM",
        "rpd": "20 RPD",
    },
    "gemma-4-31b-it": {
        "description": "Open weights con límite diario amplio",
        "rpm": "15 RPM",
        "tpm": "sin límite TPM",
        "rpd": "1.5K RPD",
    },
}
AI_MODEL_ALIASES = {
    "gemini-3-flash-preview": "gemini-3.5-flash",
    "gemma-3-27b-it": "gemma-4-31b-it",
}


def sync_mode_from_selector():
    selected_mode = (
        "figura"
        if st.session_state.mode_selector == "Modo Figura"
        else "lienzo"
    )
    st.session_state.mode = selected_mode
    if st.query_params.get("mode") != selected_mode:
        st.query_params["mode"] = selected_mode


def resolve_font_path(path):
    return path if os.path.isabs(path) else os.path.join(APP_DIR, path)


def detect_text_script(text):
    has_hiragana_katakana = False
    has_hangul = False
    has_cjk_ideograph = False
    for char in text or "":
        code = ord(char)
        if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
            has_hiragana_katakana = True
        elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
            has_hangul = True
        elif (
            0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
        ):
            has_cjk_ideograph = True
    if has_hiragana_katakana:
        return "jp"
    if has_hangul:
        return "kr"
    if has_cjk_ideograph:
        return "cjk"
    return None


def get_font_paths_for_script(script=None):
    cjk_paths = {
        "jp": [
            "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/YuGothR.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
        ],
        "kr": [
            "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/malgun.ttf",
        ],
        "cjk": [
            "/usr/share/opentype/noto/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/meiryo.ttc",
        ],
    }
    default_paths = [
        "assets/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/NotoSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    return cjk_paths.get(script, []) + default_paths

# ═══════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════

def parse_txt_file(content):
    """
    Parses the poet's TXT file into word lists and poems.
    Poems are stored as dicts: {"title": str, "content": str}
    Supports both:
        === POEMA: Título ===
        === POEMA:
    """
    lists = {}
    poems = []

    lines = content.strip().split("\n")
    current_section = None
    current_name = None
    current_content = []

    def save_current():
        """Saves whatever section we were building."""
        nonlocal current_section, current_name, current_content
        if current_section == "lista" and current_name:
            lists[current_name] = current_content
        elif current_section == "poema" and current_content:
            poems.append({
                "title": current_name or "",
                "content": "\n".join(current_content)
            })

    for line in lines:
        stripped = line.strip()

        # ── List header: === LISTA: Name === ──
        if stripped.startswith("=== LISTA:") and stripped.endswith("==="):
            save_current()
            current_section = "lista"
            current_name = stripped.replace("=== LISTA:", "").replace("===", "").strip()
            current_content = []

        # ── Poem header: === POEMA: Title === OR === POEMA: ──
        elif stripped.startswith("=== POEMA:"):
            save_current()
            current_section = "poema"
            # Extract title (works with or without closing ===)
            title_part = stripped.replace("=== POEMA:", "").replace("===", "").strip()
            current_name = title_part if title_part else None
            current_content = []

        elif stripped and current_section:
            if current_section == "lista":
                if "|" in stripped:
                    parts = stripped.split("|")
                    word = parts[0].strip()
                    translation = parts[1].strip() if len(parts) > 1 else ""
                    current_content.append({"word": word, "translation": translation})
                else:
                    current_content.append({"word": stripped, "translation": ""})
            elif current_section == "poema":
                current_content.append(stripped)

    # Save the last section
    save_current()

    return lists, poems


def get_default_font(size=20, custom_font_path=None, script=None):
    """Returns a font object with caching to avoid repeated disk reads."""
    cache_key = (size, custom_font_path, script)
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font = None

    # Try custom font first
    if custom_font_path and os.path.exists(custom_font_path):
        try:
            font = ImageFont.truetype(custom_font_path, size)
        except Exception:
            pass

    # Try bundled Noto Sans
    if font is None:
        font_paths = get_font_paths_for_script(script)
        for path in font_paths:
            path = resolve_font_path(path)
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue

    # Try DejaVu fallback
    if font is None:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            font = ImageFont.load_default()

    _font_cache[cache_key] = font
    return font


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
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()
    grayscale = image.convert("L")

    mask_array = np.array(grayscale, dtype=np.float64)
    alpha_array = np.array(a, dtype=np.float64)
    alpha_norm = alpha_array / 255.0
    result = mask_array * alpha_norm + 255.0 * (1.0 - alpha_norm)

    return Image.fromarray(result.astype(np.uint8), mode="L")


def generate_basic_shape(shape_name, width, height):
    """Generates a grayscale mask for basic geometric shapes."""
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    margin = int(min(width, height) * 0.1)

    if shape_name == "Círculo":
        draw.ellipse([margin, margin, width - margin, height - margin], fill=0)

    elif shape_name == "Corazón":
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
            points.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
        draw.polygon(points, fill=0)

    elif shape_name == "Triángulo":
        points = [(width // 2, margin), (margin, height - margin), (width - margin, height - margin)]
        draw.polygon(points, fill=0)

    elif shape_name == "Cruz":
        arm_w = min(width, height) // 4
        cx, cy = width // 2, height // 2
        half = min(width, height) // 2 - margin
        draw.rectangle([cx - arm_w//2, cy - half, cx + arm_w//2, cy + half], fill=0)
        draw.rectangle([cx - half, cy - arm_w//2, cx + half, cy + arm_w//2], fill=0)

    elif shape_name == "Óvalo":
        draw.ellipse([margin, margin + height//6, width - margin, height - margin - height//6], fill=0)

    elif shape_name == "Diamante":
        cx, cy = width // 2, height // 2
        half_w = width // 2 - margin
        half_h = height // 2 - margin
        points = [(cx, cy - half_h), (cx + half_w, cy), (cx, cy + half_h), (cx - half_w, cy)]
        draw.polygon(points, fill=0)

    elif shape_name == "Luna creciente":
        draw.ellipse([margin, margin, width - margin, height - margin], fill=0)
        offset = int(min(width, height) * 0.25)
        draw.ellipse([margin + offset, margin, width - margin + offset, height - margin], fill=255)

    return img


def prepare_background_image(bg_image, output_width, output_height, display_mode):
    """Prepares a background image according to the selected display mode."""
    img = bg_image.convert("RGBA")
    img_w, img_h = img.size

    if display_mode == "Estirar":
        return img.resize((output_width, output_height), Image.Resampling.LANCZOS)

    elif display_mode == "Rellenar":
        # Scale to cover entire canvas, crop overflow
        scale = max(output_width / img_w, output_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Center crop
        left = (new_w - output_width) // 2
        top = (new_h - output_height) // 2
        return img_scaled.crop((left, top, left + output_width, top + output_height))

    elif display_mode == "Ajustar":
        # Scale to fit entirely, letterbox with transparent
        scale = min(output_width / img_w, output_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img_scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        paste_x = (output_width - new_w) // 2
        paste_y = (output_height - new_h) // 2
        result.paste(img_scaled, (paste_x, paste_y))
        return result

    elif display_mode == "Original":
        # Center at native size
        result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        paste_x = (output_width - img_w) // 2
        paste_y = (output_height - img_h) // 2
        result.paste(img, (paste_x, paste_y))
        return result

    elif display_mode == "Mosaico":
        result = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
        for y in range(0, output_height, img_h):
            for x in range(0, output_width, img_w):
                result.paste(img, (x, y))
        return result

    # Fallback
    return img.resize((output_width, output_height), Image.Resampling.LANCZOS)


def get_text_for_rendering(parsed_lists, parsed_poems, text_source,
                           direct_text="", selected_list=None, selected_poem_idx=None,
                           preserve_edge_spaces=False):
    """Returns the final text string to be rendered, cleaned of punctuation."""
    raw_text = ""
    
    if text_source == "direct":
        raw_text = direct_text
    elif text_source == "file_poem" and parsed_poems:
        idx = selected_poem_idx if selected_poem_idx is not None else 0
        if idx < len(parsed_poems):
            raw_text = parsed_poems[idx]["content"]
    elif text_source == "file_list" and parsed_lists and selected_list:
        if selected_list in parsed_lists:
            words = [item["word"] for item in parsed_lists[selected_list]]
            raw_text = " ".join(words)
    
    if not raw_text:
        raw_text = direct_text
    
    # Clean punctuation for rendering (the poet sees formatted text,
    # but the calligram only uses the words)
    return clean_text_for_rendering(raw_text, preserve_edge_spaces=preserve_edge_spaces)


def arrange_with_gemini(word_list, style, api_key, output_format="Poema", language="", model_name="gemini-flash-latest", temperature=1.0):
    """
    Uses Gemini/Gemma to arrange indigenous words.
    Now supports dynamic model selection and temperature control.
    """
    import random
    model_name = AI_MODEL_ALIASES.get(model_name, model_name)
    
    if style == "Aleatorio":
        words_only = [item["word"] for item in word_list]
        random.shuffle(words_only)
        if output_format == "Poema":
            lines = []
            words_per_line = max(2, len(words_only) // 4)
            for i in range(0, len(words_only), words_per_line):
                lines.append(" ".join(words_only[i:i + words_per_line]))
            return "\n".join(lines), None
        else:
            return " ".join(words_only) + ".", None
    
    # Preparar el prompt (igual que antes)
    word_pairs = "\n".join(
        [f"  - {item['word']} = {item['translation']}" for item in word_list]
    )
    words_only_list = ", ".join([item["word"] for item in word_list])
    
    style_instructions = {
        "Flujo natural": (
            "Ordena las palabras indígenas en una secuencia que fluya "
            "temáticamente de manera natural y poética, como un viaje "
            "desde conceptos terrenales hacia conceptos celestiales, "
            "o desde lo concreto hacia lo abstracto. "
            "Puedes repetir palabras si eso mejora el flujo poético."
        ),
        "Contraste": (
            "Ordena las palabras indígenas alternando conceptos opuestos "
            "o contrastantes: luz/oscuridad, tierra/cielo, vida/muerte, "
            "grande/pequeño, etc. Crea un ritmo de contraste poético. "
            "Puedes repetir palabras para enfatizar los contrastes."
        ),
        "Repetición poética": (
            "Ordena las palabras indígenas creando un patrón con "
            "repeticiones deliberadas, como un estribillo o un canto. "
            "Usa la repetición de ciertas palabras clave para crear "
            "ritmo y énfasis, como en la poesía oral tradicional."
        ),
    }
    instruction = style_instructions.get(style, style_instructions["Flujo natural"])
    
    if output_format == "Poema":
        format_instruction = (
            "FORMATO DE SALIDA: Escribe el resultado como un POEMA con versos. "
            "Cada verso debe estar en una línea separada. "
            "Agrupa los versos en estrofas (separadas por una línea vacía). "
            "Puedes agregar comas y puntos entre las palabras para darle ritmo "
            "y hacer la lectura más natural. "
            "Ejemplo de formato:\n"
            "  tlalli, atl, xochitl,\n"
            "  tonatiuh ehecatl.\n"
            "\n"
            "  metztli, citlalli,\n"
            "  ilhuicatl, quiahuitl."
        )
    else:
        format_instruction = (
            "FORMATO DE SALIDA: Escribe el resultado como PROSA poética. "
            "Usa oraciones separadas por puntos. "
            "Dentro de las oraciones, usa comas para crear pausas rítmicas. "
            "Organiza el texto en 1 a 3 párrafos cortos. "
            "Ejemplo de formato:\n"
            "  Tlalli, atl, xochitl tonatiuh. "
            "Ehecatl, metztli citlalli. Ilhuicatl, quiahuitl."
        )
    
    language_context = ""
    if language.strip():
        language_context = (
            f"\nLas palabras están en lengua {language.strip()}. "
            f"Si conoces la gramática o sintaxis de esta lengua, "
            f"úsala para ordenar las palabras de forma más natural y coherente. "
            f"Si no la conoces bien, guíate por las traducciones al español.\n"
        )

    prompt = f"""Eres un asistente que ayuda a poetas indígenas a crear poesía visual (caligramas).

Aquí hay una lista de palabras en una lengua indígena mexicana con sus 
traducciones al español:
{language_context}

{word_pairs}

Tu tarea:
{instruction}

{format_instruction}

REGLAS IMPORTANTES:
1. Usa SOLAMENTE las palabras indígenas de la lista (la columna izquierda).
2. NO inventes palabras nuevas. NO agregues palabras en español ni en otro idioma.
3. Cada palabra de la lista debe aparecer AL MENOS una vez.
4. Puedes repetir palabras para mejorar el efecto poético.
5. Los únicos signos de puntuación permitidos son: , . ; ' ... 
6. No incluyas explicaciones, títulos ni notas. Solo el texto formateado.

Palabras disponibles: {words_only_list}
Tu respuesta:"""

    if "gemma" in model_name.lower():
        gemma_words = "\n".join(f"- {item['word']}" for item in word_list)
        gemma_meanings = "\n".join(
            f"- {item['word']}: {item['translation']}" for item in word_list
            if item.get("translation")
        )
        gemma_format = (
            "Use short poem lines. You may separate stanzas with blank lines."
            if output_format == "Poema"
            else "Use one or two short poetic prose sentences."
        )
        prompt = f"""Create the final {output_format.lower()} for a visual poem.

Allowed output words, exact spelling:
{gemma_words}

Meaning hints for planning only. Never output these hints:
{gemma_meanings or "(none)"}

Poetic direction:
{instruction}

Output contract:
- Output ONLY the final {output_format.lower()}.
- Every word in the output MUST be copied exactly from the allowed output words.
- Use each allowed word at least once.
- You may repeat allowed words.
- Allowed punctuation: comma, period, semicolon, apostrophe, ellipsis.
- Do not write role, task, constraints, analysis, checks, titles, bullets, Markdown, labels, translations, or explanations.

{gemma_format}

Final {output_format.lower()} only:"""

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Configuración dinámica según el modelo
        # Gemma suele tener límites de tokens de salida diferentes a los modelos Pro/Flash
        if "gemma" in model_name.lower():
            max_tokens = 8192
        else:
            max_tokens = 65536
            
        # Instanciar el modelo seleccionado por el usuario
        model = genai.GenerativeModel(model_name)
        
        def generate_text(active_prompt, active_temperature):
            response = model.generate_content(
                active_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=active_temperature,
                    max_output_tokens=max_tokens,
                )
            )
            return response.text.strip()

        result_text = generate_text(prompt, temperature)
        
        # --- VALIDACION ---
        def normalize_word(w):
            w = str(w).lower().strip()
            w = w.replace('\u2019', "'").replace('\u2018', "'").replace('\u02BC', "'")
            w = w.replace('\u2013', '-').replace('\u2014', '-')
            return w

        import re
        valid_lookup = {}
        base_word_sequence = []
        for item in word_list:
            canonical = str(item.get("word", "")).strip()
            if canonical:
                base_word_sequence.append(canonical)
                valid_lookup.setdefault(normalize_word(canonical), canonical)
        valid_words = set(valid_lookup)
        punctuation_pattern = "[.,;:!?\\u00a1\\u00bf\\\"\\u00ab\\u00bb()\\[\\]{}$/\\\\\\u2014\\u2013\\u2026\\u3001\\u3002\\n\\r\\t]"

        def split_candidate_words(text):
            check_text = re.sub(punctuation_pattern, " ", text or "")
            return check_text.split()

        def canonicalize_candidate_word(word):
            w_norm = normalize_word(word)
            if w_norm in valid_lookup:
                return valid_lookup[w_norm]
            w_cleaned = re.sub(r"^[`'\"-]+|[`'\"-]+$", "", w_norm)
            if w_cleaned in valid_lookup:
                return valid_lookup[w_cleaned]
            return None

        def candidate_sequence(text):
            sequence = []
            invalid = []
            for candidate in split_candidate_words(text):
                canonical = canonicalize_candidate_word(candidate)
                if canonical:
                    sequence.append(canonical)
                else:
                    invalid.append(candidate)
            return sequence, invalid

        def evaluate_candidate(text):
            sequence, invalid = candidate_sequence(text)
            total = len(sequence) + len(invalid)
            return len(sequence), total if total else 1

        def extract_valid_sequence(*texts):
            sequence = []
            for text in texts:
                clean_block = extract_clean_output_sequence(text)
                if clean_block:
                    sequence.extend(clean_block)
            return sequence

        def ensure_each_word_once(words):
            sequence = list(words)
            seen = {normalize_word(word) for word in sequence}
            for word in base_word_sequence:
                word_key = normalize_word(word)
                if word_key not in seen:
                    sequence.append(word)
                    seen.add(word_key)
            return sequence

        def local_fallback_sequence():
            words = list(base_word_sequence)
            if not words:
                return []
            if style == "Contraste":
                contrasted = []
                left = 0
                right = len(words) - 1
                while left <= right:
                    contrasted.append(words[left])
                    if left != right:
                        contrasted.append(words[right])
                    left += 1
                    right -= 1
                return contrasted
            if style == "Repetición poética" and len(words) > 1:
                return [words[0]] + words + [words[0], words[-1]]
            return words

        def format_safe_sequence(words):
            words = [word for word in words if word]
            if not words:
                return ""
            if output_format == "Poema":
                words_per_line = max(2, min(4, max(2, len(words) // 4)))
                lines = []
                for i in range(0, len(words), words_per_line):
                    lines.append(" ".join(words[i:i + words_per_line]))
                if len(lines) > 4:
                    midpoint = len(lines) // 2
                    return "\n".join(lines[:midpoint] + [""] + lines[midpoint:])
                return "\n".join(lines)
            words_per_sentence = max(4, min(8, max(4, len(words) // 2)))
            sentences = []
            for i in range(0, len(words), words_per_sentence):
                sentences.append(" ".join(words[i:i + words_per_sentence]) + ".")
            return " ".join(sentences)

        def contains_all_required_words(words):
            seen = {normalize_word(word) for word in words}
            return all(normalize_word(word) in seen for word in base_word_sequence)

        def dedupe_repeated_sequence(words):
            if len(words) >= 4 and len(words) % 2 == 0:
                midpoint = len(words) // 2
                if words[:midpoint] == words[midpoint:]:
                    return words[:midpoint]
            return words

        def line_is_clean_output(line):
            stripped = line.strip()
            if not stripped:
                return False
            if any(marker in stripped for marker in ("*", "`", "#", "[", "]", "<", ">", "=")):
                return False
            sequence, invalid = candidate_sequence(stripped)
            return bool(sequence) and not invalid

        def extract_clean_output_sequence(text):
            blocks = []
            current = []
            for raw_line in str(text or "").splitlines():
                line = raw_line.strip()
                if not line:
                    if current and current[-1] != "":
                        current.append("")
                    continue
                if line_is_clean_output(line):
                    current.append(line)
                elif current:
                    blocks.append(current)
                    current = []
            if current:
                blocks.append(current)

            candidates = []
            for block in blocks:
                while block and block[-1] == "":
                    block.pop()
                if not block:
                    continue
                sequence, invalid = candidate_sequence("\n".join(block))
                if sequence and not invalid:
                    candidates.append(sequence)

            if not candidates:
                sequence, invalid = candidate_sequence(text)
                if sequence and not invalid:
                    return dedupe_repeated_sequence(sequence)
                return []

            complete = [seq for seq in candidates if contains_all_required_words(seq)]
            chosen = complete[-1] if complete else candidates[-1]
            return dedupe_repeated_sequence(chosen)

        def safe_gemma_output(*texts):
            for text in texts:
                clean_sequence = extract_clean_output_sequence(text)
                if clean_sequence and contains_all_required_words(clean_sequence):
                    return format_safe_sequence(dedupe_repeated_sequence(clean_sequence))

            extracted = extract_valid_sequence(*texts)
            if extracted:
                return format_safe_sequence(ensure_each_word_once(dedupe_repeated_sequence(extracted)))
            return format_safe_sequence(local_fallback_sequence())

        matching, total_check = evaluate_candidate(result_text)

        if "gemma" in model_name.lower():
            clean_result = extract_clean_output_sequence(result_text)
            if clean_result and contains_all_required_words(clean_result):
                return format_safe_sequence(clean_result), None

            strict_retry_prompt = prompt + f"""

RETRY: Your previous answer used words outside the allowed list.
Write again using ONLY these exact words: {words_only_list}.
Return only final poem/prose lines. No analysis, bullets, labels, checks, or Markdown."""
            retry_text = generate_text(strict_retry_prompt, min(temperature, 0.7))
            clean_retry = extract_clean_output_sequence(retry_text)
            if clean_retry and contains_all_required_words(clean_retry):
                return format_safe_sequence(clean_retry), None
            return safe_gemma_output(retry_text, result_text), None
        
        # Si la coincidencia es muy baja, lanzamos advertencia pero devolvemos lo que hay
        if matching < total_check * 0.3:
            return None, (
                f"⚠️ El modelo {model_name} generó texto, pero inventó demasiadas palabras. "
                f"Prueba bajando la 'Creatividad' (temperatura) o cambiando de modelo."
            )
        
        return result_text, None
        
    except Exception as e:
        return None, f"❌ Error ({model_name}): {str(e)}"


def clean_text_for_rendering(text, preserve_edge_spaces=False):
    """
    Removes punctuation and formatting from text before rendering.
    Preserves special characters used in indigenous Mexican languages:
    - Apostrophes/glottal stops within words (k'iin, ts'o'ok, ch')
    - Hyphens within compound words (ni-k-tlazohtla)
    - Colons within words for vowel length (a:, tuka:ri)
    - Glottal stop character ʔ (treated as a letter)
    - Modifier apostrophe ʼ (treated as a letter)
    - All accented, nasalized, and special Unicode letters (á, ñ, ü, ã, ǎ, etc.)
    """
    import re
    original_text = text or ""
    
    # Step 1: Normalize line breaks to spaces
    cleaned = original_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Step 1.5: Normalize apostrophe and dash variants
    # (so indigenous words with glottal stops are consistent)
    cleaned = cleaned.replace('\u2019', "'")  # right single quotation mark → straight
    cleaned = cleaned.replace('\u2018', "'")  # left single quotation mark → straight
    cleaned = cleaned.replace('\u02BC', "'")  # modifier letter apostrophe → straight
    cleaned = cleaned.replace('\u02BB', "'")  # modifier letter turned comma → straight
    cleaned = cleaned.replace('\u2013', '-')  # en dash → hyphen
    cleaned = cleaned.replace('\u2014', '-')  # em dash → hyphen
    cleaned = cleaned.replace('\u2010', '-')  # hyphen character → standard hyphen
    cleaned = cleaned.replace('\u2011', '-')  # non-breaking hyphen → standard hyphen
    
    # Step 2: Remove unambiguous punctuation (never part of indigenous words)
    cleaned = re.sub(r'[.,;!?¡¿\u201c\u201d\u00ab\u00bb()$$$${}/\\—–…\u201c\u201d\u2022\u00b7°]', ' ', cleaned)
    
    # Step 3: Handle apostrophes and glottal stop markers
    # KEEP when adjacent to word characters: k'iin, ts'o'ok, ch'
    # REMOVE only when isolated (no word character on either side)
    cleaned = re.sub(r"(?<!\w)['''ʼ](?!\w)", ' ', cleaned)
    
    # Step 4: Handle hyphens — keep between word characters
    # Preserves: ni-k-tlazohtla    Removes: word - word, — (dash)
    cleaned = re.sub(r"(?<!\w)-(?!\w)", ' ', cleaned)
    
    # Step 5: Handle colons — keep between word characters (vowel length)
    # Preserves: tuka:ri, a:    Removes: Título: texto
    cleaned = re.sub(r"(?<!\w):(?!\w)", ' ', cleaned)
    
    # Step 6: Collapse multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()

    if preserve_edge_spaces and cleaned:
        if original_text[:1].isspace():
            cleaned = " " + cleaned
        if original_text[-1:].isspace():
            cleaned = cleaned + " "

    return cleaned


def make_lienzo_text_sources(parsed_lists, parsed_poems, current_text):
    """Builds up to ten selectable text sources for the browser canvas."""
    sources = []
    seen = set()

    def add_source(label, value):
        cleaned = clean_text_for_rendering(value or "", preserve_edge_spaces=(label == "Texto actual"))
        if not cleaned.strip():
            return
        key = (label, cleaned)
        if key in seen:
            return
        seen.add(key)
        sources.append({"label": label, "text": cleaned})

    add_source("Texto actual", current_text)

    for i, poem in enumerate(parsed_poems or []):
        title = poem.get("title") or f"Poema {i + 1}"
        add_source(f"Poema: {title}", poem.get("content", ""))

    for name, items in (parsed_lists or {}).items():
        words = " ".join(item.get("word", "") for item in items)
        add_source(f"Lista: {name}", words)

    while len(sources) < 10 and sources:
        sources.append({"label": f"Texto {len(sources) + 1}", "text": sources[0]["text"]})

    return sources[:10]


def get_lienzo_font_payload(text=""):
    """Returns font metadata for the browser canvas."""
    custom_bytes = st.session_state.get("custom_font_bytes")
    if custom_bytes:
        return {
            "family": "XquendArtCanvasFont",
            "data": base64.b64encode(custom_bytes).decode("ascii"),
            "stack": '"XquendArtCanvasFont", "Noto Sans", "Noto Sans CJK JP", Arial, sans-serif',
        }

    script = detect_text_script(text)
    font_path = None
    for candidate in get_font_paths_for_script(script):
        resolved = resolve_font_path(candidate)
        if os.path.exists(resolved):
            font_path = resolved
            break

    stack = '"Noto Sans", "Noto Sans CJK JP", "Noto Sans CJK KR", "Noto Sans CJK SC", "Yu Gothic", "Meiryo", "Malgun Gothic", "Microsoft YaHei", Arial, sans-serif'
    if font_path and os.path.getsize(font_path) <= 2_500_000:
        with open(font_path, "rb") as font_file:
            return {
                "family": "XquendArtCanvasFont",
                "data": base64.b64encode(font_file.read()).decode("ascii"),
                "stack": f'"XquendArtCanvasFont", {stack}',
            }

    return {"family": "XquendArtCanvasFont", "data": None, "stack": stack}


def build_lienzo_canvas_html(config):
    """Builds the self-contained HTML/JS word-painting canvas."""
    template_path = os.path.join(APP_DIR, "assets", "lienzo_canvas_template.html")
    with open(template_path, "r", encoding="utf-8") as template_file:
        template = template_file.read()
    config_json = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__XQUENDART_CONFIG__", config_json)


def add_title_and_author(image, svg_string,
                         title_text="", author_text="",
                         title_font_size=48, author_font_size=24,
                         title_position="top-center", author_position="bottom-right",
                         title_color="#000000", author_color="#000000",
                         custom_font_path=None):
    """
    Draws a title and an author signature on top of the finished calligram.
    Title is ALWAYS rendered visually above the author when they share the
    same edge (top or bottom), regardless of corner.
    """
    title = title_text.strip()
    author = author_text.strip()

    if not title and not author:
        return image, svg_string

    output = image.copy()
    draw = ImageDraw.Draw(output)

    img_w, img_h = output.size
    margin = int(min(img_w, img_h) * 0.005)

    # ──────────────────────────────────────────────────────────
    # VERTICAL GAP between title and author when on the same edge.
    # Change this value to increase or decrease the separation.
    # ──────────────────────────────────────────────────────────
    title_author_gap = margin  # ← MODIFY THIS for more/less separation

    svg_additions = []

    def hex_to_rgba(hex_color):
        return (
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
            255
        )

    def measure_text(text_str, font):
        try:
            bbox = font.getbbox(text_str)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            return int(font.size * len(text_str) * 0.6), int(font.size * 1.2)

    def compute_xy(position, text_w, text_h, vertical_offset=0):
        if "center" in position:
            x = (img_w - text_w) // 2
        elif "left" in position:
            x = margin
        else:
            x = img_w - text_w - margin

        if "top" in position:
            y = margin + vertical_offset
        else:
            y = img_h - text_h - margin - vertical_offset

        return x, y

    def same_edge(pos_a, pos_b):
        return ("top" in pos_a) == ("top" in pos_b)

    # ── Measure both elements first ──
    title_font = author_font = None
    tc = ac = None
    tw = th = aw = ah = 0

    if title:
        title_font = get_default_font(title_font_size, custom_font_path)
        tc = hex_to_rgba(title_color)
        tw, th = measure_text(title, title_font)

    if author:
        author_font = get_default_font(author_font_size, custom_font_path)
        ac = hex_to_rgba(author_color)
        aw, ah = measure_text(author, author_font)

    # ── Determine shared-edge situation ──
    shared_edge = title and author and same_edge(title_position, author_position)
    on_bottom = shared_edge and "bottom" in title_position

    # ── Draw Title ──
    if title:
        title_offset = 0
        if shared_edge and on_bottom:
            # Bottom edge: push TITLE up so it sits above the author
            title_offset = ah + title_author_gap
        
        tx, ty = compute_xy(title_position, tw, th, vertical_offset=title_offset)
        draw.text((tx, ty), title, fill=tc, font=title_font)

        escaped = html.escape(title)
        svg_additions.append(
            f'<text x="{tx}" y="{ty + title_font_size}" '
            f'font-size="{title_font_size}" '
            f'fill="rgb({tc[0]},{tc[1]},{tc[2]})" '
            f'font-family="Noto Sans, Noto Sans CJK JP, Noto Sans CJK KR, Noto Sans CJK SC, sans-serif">'
            f'{escaped}</text>'
        )

    # ── Draw Author ──
    if author:
        author_offset = 0
        if shared_edge and not on_bottom:
            # Top edge: push AUTHOR down so it sits below the title
            author_offset = th + title_author_gap

        ax, ay = compute_xy(author_position, aw, ah, vertical_offset=author_offset)
        draw.text((ax, ay), author, fill=ac, font=author_font)

        escaped = html.escape(author)
        svg_additions.append(
            f'<text x="{ax}" y="{ay + author_font_size}" '
            f'font-size="{author_font_size}" '
            f'fill="rgb({ac[0]},{ac[1]},{ac[2]})" '
            f'font-family="Noto Sans, Noto Sans CJK JP, Noto Sans CJK KR, Noto Sans CJK SC, sans-serif">'
            f'{escaped}</text>'
        )

    # ── Inject into SVG ──
    if svg_additions and svg_string:
        closing_tag = "</svg>"
        additions_str = "\n".join(f"  {elem}" for elem in svg_additions)
        svg_string = svg_string.replace(
            closing_tag, f"{additions_str}\n{closing_tag}"
        )

    return output, svg_string


# ═══════════════════════════════════════════
# CORE RENDERING ENGINE
# ═══════════════════════════════════════════

def fill_shape_with_text(mask, text, config):
    """
    Core rendering engine: fills a grayscale mask with text.
    Texture mode uses occupancy grid. Silueta/Contorno use scanline.
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
    custom_font_path = config.get("custom_font_path", None)
    text_script = detect_text_script(text)

    ABSOLUTE_MIN_FONT = 4
    BACKGROUND_THRESHOLD = 253  # pixels > this are background

    # ── Prepare the mask at output resolution ──
    mask_full = mask.resize((output_width, output_height), Image.Resampling.LANCZOS)
    mask_array = np.array(mask_full)

    if invert:
        mask_array = 255 - mask_array

    # ── Prepare output image ──
    bg_image = config.get("bg_image", None)
    bg_display_mode = config.get("bg_display_mode", "Rellenar")

    if bg_image is not None:
        output = prepare_background_image(bg_image, output_width, output_height, bg_display_mode)
        # Ensure RGBA
        if output.mode != "RGBA":
            output = output.convert("RGBA")
    elif bg_transparent:
        output = Image.new("RGBA", (output_width, output_height), (0, 0, 0, 0))
    else:
        bg_r = int(bg_color[1:3], 16)
        bg_g = int(bg_color[3:5], 16)
        bg_b = int(bg_color[5:7], 16)
        output = Image.new("RGBA", (output_width, output_height), (bg_r, bg_g, bg_b, 255))

    draw = ImageDraw.Draw(output)

    tc_r = int(text_color[1:3], 16)
    tc_g = int(text_color[3:5], 16)
    tc_b = int(text_color[5:7], 16)
    text_fill = (tc_r, tc_g, tc_b, 255)

    words = text.split()
    if not words:
        return output, ""

    svg_elements = []

    # ── Helper: add SVG element ──
    def add_svg(x, y, font_size, word_text):
        svg_elements.append(
            f'<text x="{x}" y="{y + font_size}" '
            f'font-size="{font_size}" '
            f'fill="rgb({tc_r},{tc_g},{tc_b})" '
            f'font-family="Noto Sans, Noto Sans CJK JP, Noto Sans CJK KR, Noto Sans CJK SC, sans-serif">'
            f'{word_text}</text>'
        )

    # ══════════════════════════════════════════════════
    # TEXTURE MODE — Occupancy Grid Approach
    # ══════════════════════════════════════════════════
    if mode == "textura":
        # Identify subject pixels
        subject_mask = mask_array <= BACKGROUND_THRESHOLD

        if not np.any(subject_mask):
            return output, ""

        # Normalize subject brightness range to 0-255
        subject_pixels = mask_array[subject_mask]
        val_min = float(np.percentile(subject_pixels, 2))
        val_max = float(np.percentile(subject_pixels, 98))

        if val_max - val_min < 5:
            val_min = float(subject_pixels.min())
            val_max = float(subject_pixels.max())
        if val_max - val_min < 1:
            val_max = val_min + 1

        # Create normalized brightness array
        norm_array = np.full((output_height, output_width), 255.0, dtype=np.float64)
        norm_array[subject_mask] = np.clip(
            (mask_array[subject_mask].astype(np.float64) - val_min)
            / (val_max - val_min) * 255.0,
            0, 255
        )

        # Occupancy grid — tracks which pixels already have text
        occupied = np.zeros((output_height, output_width), dtype=bool)

        # Scan step: higher density = smaller step = finer detail
        step = max(2, int(min(output_width, output_height) / density))

        word_index = 0

        def darkness_to_font(pixel_val):
            """Maps normalized pixel (0=black, 255=white) to font size."""
            darkness = 1.0 - (pixel_val / 255.0)
            darkness = max(0.0, min(1.0, darkness))
            # Power curve for more dramatic contrast
            darkness = darkness ** 0.7
            size = int(font_min + (font_max - font_min) * darkness)
            return max(ABSOLUTE_MIN_FONT, min(font_max, size))

        def check_occupancy(x, y, w, h):
            """Returns True if the rectangle area is free (unoccupied)."""
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(output_width, int(x + w))
            y2 = min(output_height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                return False
            return not np.any(occupied[y1:y2, x1:x2])

        def mark_occupied(x, y, w, h):
            """Marks rectangle + spacing margin as occupied."""
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(output_width, int(x + w + spacing))
            y2 = min(output_height, int(y + h + spacing))
            occupied[y1:y2, x1:x2] = True

        def is_within_subject(x, y, w, h):
            """Checks that the word stays mostly within the subject area."""
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(output_width, int(x + w))
            y2 = min(output_height, int(y + h))
            if x2 <= x1 or y2 <= y1:
                return False
            region = subject_mask[y1:y2, x1:x2]
            if region.size == 0:
                return False
            return np.mean(region) > 0.5

        # ── Determine scan order based on direction ──
        if direction == "lr":
            scan_positions = [
                (sy, sx)
                for sy in range(0, output_height, step)
                for sx in range(0, output_width, step)
            ]
        elif direction == "rl":
            scan_positions = [
                (sy, sx)
                for sy in range(0, output_height, step)
                for sx in range(output_width - 1, -1, -step)
            ]
        elif direction == "tb":
            scan_positions = [
                (sy, sx)
                for sx in range(0, output_width, step)
                for sy in range(0, output_height, step)
            ]
        elif direction == "bt":
            scan_positions = [
                (sy, sx)
                for sx in range(0, output_width, step)
                for sy in range(output_height - 1, -1, -step)
            ]
        else:
            scan_positions = [
                (sy, sx)
                for sy in range(0, output_height, step)
                for sx in range(0, output_width, step)
            ]

        # ── Main placement loop ──
        for scan_y, scan_x in scan_positions:
            # Bounds check
            if scan_y >= output_height or scan_x >= output_width:
                continue
            if scan_y < 0 or scan_x < 0:
                continue

            # Skip background pixels
            if not subject_mask[scan_y, scan_x]:
                continue

            # Skip already occupied pixels
            if occupied[scan_y, scan_x]:
                continue

            # Stop if we've run out of words and looping is off
            if not loop_text and word_index >= len(words):
                break

            # Get normalized pixel value and compute ideal font size
            pixel_val = norm_array[scan_y, scan_x]
            ideal_fs = darkness_to_font(pixel_val)

            # Get next word
            word = words[word_index % len(words)]

            # Try to place at ideal size, shrink if necessary
            placed = False
            fs = ideal_fs

            while fs >= ABSOLUTE_MIN_FONT:
                font = get_default_font(fs, custom_font_path, text_script)
                try:
                    bbox = font.getbbox(word)
                    w_w = bbox[2] - bbox[0]
                    w_h = bbox[3] - bbox[1]
                except Exception:
                    w_w = fs * len(word) * 0.6
                    w_h = fs * 1.2

                # Check if it fits: unoccupied AND within subject
                if (check_occupancy(scan_x, scan_y, w_w, w_h) and
                        is_within_subject(scan_x, scan_y, w_w, w_h)):
                    # Place the word
                    draw.text((scan_x, scan_y), word, fill=text_fill, font=font)
                    add_svg(scan_x, scan_y, fs, word)
                    mark_occupied(scan_x, scan_y, w_w, w_h)
                    word_index += 1
                    placed = True
                    break

                # Shrink and try again
                fs -= 1

            # Last resort: if couldn't place even at ABSOLUTE_MIN,
            # force-place at minimum size to avoid blank spots
            if not placed:
                fs = ABSOLUTE_MIN_FONT
                font = get_default_font(fs, custom_font_path, text_script)
                try:
                    bbox = font.getbbox(word)
                    w_w = bbox[2] - bbox[0]
                    w_h = bbox[3] - bbox[1]
                except Exception:
                    w_w = fs * len(word) * 0.6
                    w_h = fs * 1.2

                # Force place even if overlapping — no blank spots
                draw.text((scan_x, scan_y), word, fill=text_fill, font=font)
                add_svg(scan_x, scan_y, fs, word)
                mark_occupied(scan_x, scan_y, w_w, w_h)
                word_index += 1

        # ── Second pass: fill remaining gaps within subject ──
        # Scan at finer resolution to catch small unfilled areas
        fine_step = max(2, step // 2)
        for scan_y in range(0, output_height, fine_step):
            for scan_x in range(0, output_width, fine_step):
                if scan_y >= output_height or scan_x >= output_width:
                    continue
                if not subject_mask[scan_y, scan_x]:
                    continue
                if occupied[scan_y, scan_x]:
                    continue

                if not loop_text and word_index >= len(words):
                    break

                word = words[word_index % len(words)]
                fs = ABSOLUTE_MIN_FONT
                font = get_default_font(fs, custom_font_path, text_script)

                try:
                    bbox = font.getbbox(word)
                    w_w = bbox[2] - bbox[0]
                    w_h = bbox[3] - bbox[1]
                except Exception:
                    w_w = fs * len(word) * 0.6
                    w_h = fs * 1.2

                if check_occupancy(scan_x, scan_y, w_w, w_h):
                    draw.text((scan_x, scan_y), word, fill=text_fill, font=font)
                    add_svg(scan_x, scan_y, fs, word)
                    mark_occupied(scan_x, scan_y, w_w, w_h)
                    word_index += 1

            if not loop_text and word_index >= len(words):
                break

    # ══════════════════════════════════════════════════
    # SILUETA AND CONTORNO MODES — Scanline Approach
    # (unchanged from previous version)
    # ══════════════════════════════════════════════════
    elif mode in ("silueta", "contorno"):
        word_index = 0

        # ── Build edge map for contorno mode ──
        edge_map = None
        if mode == "contorno":
            mask_small = mask.resize((density, density), Image.Resampling.LANCZOS)
            small_array = np.array(mask_small)
            if invert:
                small_array = 255 - small_array

            # Subject = dark pixels (inside the shape)
            subject = small_array < 240

            # Edge = subject pixels that border non-subject pixels
            eroded = ndimage.binary_erosion(subject)
            edge_small = subject & ~eroded

            # Dilate edges to make them thick enough for text (3 iterations)
            # Only dilate within the subject area to prevent bleeding outside
            edge_small = ndimage.binary_dilation(edge_small, iterations=3) & subject

            # Scale to output resolution
            edge_img = Image.fromarray(edge_small.astype(np.uint8) * 255, mode="L")
            edge_full = edge_img.resize((output_width, output_height), Image.Resampling.NEAREST)
            edge_map = np.array(edge_full) > 127

        # ── Helper: font size for silueta/contorno ──
        def get_font_size_scanline(current_mode, max_cell_size):
            if current_mode == "contorno":
                size = int(font_min + (font_max - font_min) * 0.4)
            else:
                size = int(font_min + (font_max - font_min) * 0.6)
            size = max(font_min, min(font_max, size))
            max_allowed = int(max_cell_size * 0.92)
            if max_allowed >= ABSOLUTE_MIN_FONT:
                size = min(size, max_allowed)
            else:
                size = ABSOLUTE_MIN_FONT
            return size

        # ══════════════════════════════════════
        # HORIZONTAL SCANNING (L→R and R→L)
        # ══════════════════════════════════════
        if direction in ("lr", "rl"):
            row_height_base = output_height / density
            rows = list(range(density))

            for row_idx in rows:
                y = int(row_idx * row_height_base)
                if y >= output_height:
                    continue

                sample_y = min(y + int(row_height_base / 2), output_height - 1)

                if mode == "contorno":
                    row_mask = edge_map[sample_y, :] if edge_map is not None else np.zeros(output_width, dtype=bool)
                else:
                    row_mask = mask_array[sample_y, :] < 240

                if not np.any(row_mask):
                    continue

                segments = []
                in_segment = False
                seg_start = 0
                for x in range(output_width):
                    if row_mask[x] and not in_segment:
                        seg_start = x
                        in_segment = True
                    elif not row_mask[x] and in_segment:
                        segments.append((seg_start, x))
                        in_segment = False
                if in_segment:
                    segments.append((seg_start, output_width))

                if mode == "contorno" and segments:
                    merge_threshold = int(output_width / density * 2)
                    merged = [segments[0]]
                    for seg in segments[1:]:
                        if seg[0] - merged[-1][1] < merge_threshold:
                            merged[-1] = (merged[-1][0], seg[1])
                        else:
                            merged.append(seg)
                    segments = merged

                if direction == "rl":
                    segments = segments[::-1]

                for seg_start, seg_end in segments:
                    seg_width = seg_end - seg_start
                    if seg_width < 15:
                        continue

                    layout_words = []
                    temp_x = 0
                    temp_word_idx = word_index

                    while True:
                        if not loop_text and temp_word_idx >= len(words):
                            break

                        word = words[temp_word_idx % len(words)]

                        fs = get_font_size_scanline(mode, row_height_base)
                        font = get_default_font(fs, custom_font_path, text_script)

                        try:
                            bbox = font.getbbox(word)
                            w_width = bbox[2] - bbox[0]
                        except Exception:
                            w_width = fs * len(word) * 0.6

                        if temp_x + w_width > seg_width and layout_words:
                            break

                        if temp_x + w_width > seg_width and not layout_words:
                            fs = max(ABSOLUTE_MIN_FONT, int(fs * seg_width / max(w_width, 1)))
                            font = get_default_font(fs, custom_font_path, text_script)
                            try:
                                bbox = font.getbbox(word)
                                w_width = bbox[2] - bbox[0]
                            except Exception:
                                w_width = fs * len(word) * 0.6
                            if temp_x + w_width > seg_width:
                                break

                        layout_words.append({"word": word, "font_size": fs, "width": w_width})
                        temp_x += w_width + spacing
                        temp_word_idx += 1

                    if not layout_words:
                        continue

                    word_index = temp_word_idx

                    total_words_width = sum(w["width"] for w in layout_words)
                    num_gaps = len(layout_words) - 1

                    if num_gaps > 0 and mode != "contorno":
                        remaining_space = seg_width - total_words_width
                        justified_gap = max(spacing, remaining_space / num_gaps)
                        max_gap = seg_width * 0.08
                        justified_gap = min(justified_gap, max_gap)
                    else:
                        justified_gap = spacing

                    if num_gaps > 0 and mode != "contorno":
                        actual_total = total_words_width + justified_gap * num_gaps
                        start_offset = (seg_width - actual_total) / 2
                        start_offset = max(0, start_offset)
                    else:
                        start_offset = spacing

                    if direction == "rl":
                        render_x = seg_end - start_offset
                        for lw in layout_words:
                            render_x -= lw["width"]
                            font = get_default_font(lw["font_size"], custom_font_path, text_script)
                            draw_y = y + (row_height_base - lw["font_size"]) / 2
                            draw.text((render_x, draw_y), lw["word"], fill=text_fill, font=font)
                            add_svg(render_x, draw_y, lw["font_size"], lw["word"])
                            render_x -= justified_gap
                    else:
                        render_x = seg_start + start_offset
                        for lw in layout_words:
                            font = get_default_font(lw["font_size"], custom_font_path, text_script)
                            draw_y = y + (row_height_base - lw["font_size"]) / 2
                            draw.text((render_x, draw_y), lw["word"], fill=text_fill, font=font)
                            add_svg(render_x, draw_y, lw["font_size"], lw["word"])
                            render_x += lw["width"] + justified_gap

        # ══════════════════════════════════════
        # VERTICAL SCANNING (T→B and B→T)
        # ══════════════════════════════════════
        elif direction in ("tb", "bt"):
            col_width_base = output_width / density
            columns = list(range(density))

            for col_idx in columns:
                x = int(col_idx * col_width_base)
                if x >= output_width:
                    continue

                sample_x = min(x + int(col_width_base / 2), output_width - 1)

                if mode == "contorno":
                    col_mask = edge_map[:, sample_x] if edge_map is not None else np.zeros(output_height, dtype=bool)
                else:
                    col_mask = mask_array[:, sample_x] < 240

                if not np.any(col_mask):
                    continue

                segments = []
                in_segment = False
                seg_start = 0
                for y_scan in range(output_height):
                    if col_mask[y_scan] and not in_segment:
                        seg_start = y_scan
                        in_segment = True
                    elif not col_mask[y_scan] and in_segment:
                        segments.append((seg_start, y_scan))
                        in_segment = False
                if in_segment:
                    segments.append((seg_start, output_height))

                if mode == "contorno" and segments:
                    merge_threshold = int(output_height / density * 2)
                    merged = [segments[0]]
                    for seg in segments[1:]:
                        if seg[0] - merged[-1][1] < merge_threshold:
                            merged[-1] = (merged[-1][0], seg[1])
                        else:
                            merged.append(seg)
                    segments = merged

                if direction == "bt":
                    segments = segments[::-1]

                for seg_top, seg_bottom in segments:
                    seg_height = seg_bottom - seg_top
                    if seg_height < 15:
                        continue

                    layout_words = []
                    temp_y = 0
                    temp_word_idx = word_index

                    while True:
                        if not loop_text and temp_word_idx >= len(words):
                            break

                        word = words[temp_word_idx % len(words)]

                        fs = get_font_size_scanline(mode, col_width_base)
                        font = get_default_font(fs, custom_font_path, text_script)

                        try:
                            bbox = font.getbbox(word)
                            w_width = bbox[2] - bbox[0]
                            w_height = bbox[3] - bbox[1]
                        except Exception:
                            w_width = fs * len(word) * 0.6
                            w_height = fs * 1.2

                        if w_width > col_width_base * 0.95:
                            fs_try = int(fs * (col_width_base * 0.90) / max(w_width, 1))
                            fs_try = max(ABSOLUTE_MIN_FONT, fs_try)
                            font = get_default_font(fs_try, custom_font_path, text_script)
                            try:
                                bbox = font.getbbox(word)
                                w_width = bbox[2] - bbox[0]
                                w_height = bbox[3] - bbox[1]
                            except Exception:
                                w_width = fs_try * len(word) * 0.6
                                w_height = fs_try * 1.2
                            fs = fs_try

                        if w_width > col_width_base * 1.1:
                            temp_word_idx += 1
                            continue

                        if temp_y + w_height > seg_height and layout_words:
                            break

                        if temp_y + w_height > seg_height and not layout_words:
                            fs = ABSOLUTE_MIN_FONT
                            font = get_default_font(fs, custom_font_path, text_script)
                            try:
                                bbox = font.getbbox(word)
                                w_width = bbox[2] - bbox[0]
                                w_height = bbox[3] - bbox[1]
                            except Exception:
                                w_width = fs * len(word) * 0.6
                                w_height = fs * 1.2
                            if temp_y + w_height > seg_height:
                                break

                        layout_words.append({
                            "word": word, "font_size": fs,
                            "width": w_width, "height": w_height,
                        })
                        temp_y += w_height + spacing
                        temp_word_idx += 1

                    if not layout_words:
                        continue

                    word_index = temp_word_idx

                    total_words_height = sum(w["height"] for w in layout_words)
                    num_gaps = len(layout_words) - 1

                    if num_gaps > 0 and mode != "contorno":
                        remaining_space = seg_height - total_words_height
                        justified_gap = max(spacing, remaining_space / num_gaps)
                        max_gap = seg_height * 0.08
                        justified_gap = min(justified_gap, max_gap)
                    else:
                        justified_gap = spacing

                    if num_gaps > 0 and mode != "contorno":
                        actual_total = total_words_height + justified_gap * num_gaps
                        start_offset = (seg_height - actual_total) / 2
                        start_offset = max(0, start_offset)
                    else:
                        start_offset = spacing

                    if direction == "bt":
                        render_y = seg_bottom - start_offset
                        for lw in layout_words:
                            render_y -= lw["height"]
                            font = get_default_font(lw["font_size"], custom_font_path, text_script)
                            draw_x = x + (col_width_base - lw["width"]) / 2
                            draw.text((draw_x, render_y), lw["word"], fill=text_fill, font=font)
                            add_svg(draw_x, render_y, lw["font_size"], lw["word"])
                            render_y -= justified_gap
                    else:
                        render_y = seg_top + start_offset
                        for lw in layout_words:
                            font = get_default_font(lw["font_size"], custom_font_path, text_script)
                            draw_x = x + (col_width_base - lw["width"]) / 2
                            draw.text((draw_x, render_y), lw["word"], fill=text_fill, font=font)
                            add_svg(draw_x, render_y, lw["font_size"], lw["word"])
                            render_y += lw["height"] + justified_gap

    # ── Build SVG string ──
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
    st.radio(
        "Selecciona el modo:",
        ["Modo Figura", "Modo Lienzo"],
        label_visibility="collapsed",
        key="mode_selector",
        on_change=sync_mode_from_selector,
    )
    sync_mode_from_selector()

    st.divider()

    # ── Text Source ──
    st.subheader("Fuente de Texto")
    text_source = st.radio(
        "¿De dónde viene el texto?",
        ["Escribir directamente", "Subir archivo TXT", "Ordenar con IA"],
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
            placeholder="Escribe aquí las palabras o el poema...",
            key="direct_text_value"
        )

    elif text_source == "Subir archivo TXT":
        uploaded_txt = st.file_uploader(
            "Sube tu archivo TXT:",
            type=["txt"],
            help="Formato: === LISTA: Nombre === y === POEMA: Título ==="
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

            source_options = []
            if parsed_poems:
                for i, poem in enumerate(parsed_poems):
                    title = poem.get("title", "")
                    if title:
                        source_options.append(f"Poema {i+1}: {title}")
                    else:
                        preview = poem["content"][:50]
                        if len(poem["content"]) > 50:
                            preview += "..."
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

    elif text_source == "Ordenar con IA":
            st.session_state.text_source = "gemini"
            
            api_key = st.text_input(
                "Clave API de Google AI Studio:",
                type="password",
                help="Tu clave personal de Google AI Studio."
            )
            
            col_model, col_temp = st.columns([2, 1])
            
            with col_model:
                gemini_model = st.selectbox(
                    "Modelo de IA:",
                    AI_MODEL_OPTIONS,
                    help="Si tienes errores de límite (quota), prueba gemini-3.1-flash-lite o Gemma."
                )
                
            with col_temp:
                # SLIDER DE TEMPERATURA
                temperature = st.slider(
                    "Creatividad:",
                    min_value=0.0,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="0.0: Predecible/Repetitivo. 2.0: Caótico/Original."
                )
                
            model_limits = AI_MODEL_LIMITS.get(gemini_model, {})
            st.caption(
                f"ℹ️ **{gemini_model}:** {model_limits.get('description', 'Modelo disponible')}. "
                f"Límites gratuitos: {model_limits.get('rpm', 'RPM no especificado')}, "
                f"{model_limits.get('tpm', 'TPM no especificado')}, "
                f"{model_limits.get('rpd', 'RPD no especificado')}."
            )
    
            poet_language = st.text_input(
                "Lengua indígena:",
                placeholder="Ej: Náhuatl, Maya, Zapoteco...",
            )
            
            col_style, col_format = st.columns(2)
            with col_style:
                arrangement_style = st.selectbox(
                    "Estilo:",
                    ["Flujo natural", "Contraste", "Repetición poética", "Aleatorio"]
                )
            with col_format:
                output_format = st.radio(
                    "Salida:", ["Poema", "Prosa"], horizontal=True
                )
            
            st.markdown("---")
            st.markdown("**Palabras para ordenar:**")
            
            gemini_word_source = st.radio(
                "Fuente de palabras:",
                ["Escribir lista manualmente", "Usar lista de archivo TXT"],
                label_visibility="collapsed"
            )
            
            # --- PERSISTENCIA DE LISTA (Igual que antes) ---
            if "gemini_word_list_persistent" not in st.session_state:
                st.session_state.gemini_word_list_persistent = []
    
            if gemini_word_source == "Escribir lista manualmente":
                gemini_raw = st.text_area(
                    "Una palabra por línea (palabra | traducción):",
                    height=100
                )
                if gemini_raw.strip():
                    temp_list = []
                    for line in gemini_raw.strip().split("\n"):
                        if "|" in line:
                            parts = line.split("|")
                            temp_list.append({"word": parts[0].strip(), "translation": parts[1].strip()})
                        elif line.strip():
                            temp_list.append({"word": line.strip(), "translation": ""})
                    st.session_state.gemini_word_list_persistent = temp_list
                            
            elif gemini_word_source == "Usar lista de archivo TXT":
                gemini_txt = st.file_uploader(
                    "Sube archivo TXT:", type=["txt"], key="gemini_txt_uploader"
                )
                if gemini_txt is not None:
                    content = gemini_txt.read().decode("utf-8")
                    g_lists, _ = parse_txt_file(content)
                    if g_lists:
                        chosen_list = st.selectbox("Elegir lista:", list(g_lists.keys()))
                        st.session_state.gemini_word_list_persistent = g_lists[chosen_list]
                        st.caption(f"Vista: {', '.join([x['word'] for x in st.session_state.gemini_word_list_persistent[:4]])}...")
    
            current_word_list = st.session_state.gemini_word_list_persistent
            
            if current_word_list:
                st.info(f"📝 {len(current_word_list)} palabras cargadas.")
                
                # BOTÓN DE ORDENAR
                if st.button("✨ Ordenar palabras con IA", use_container_width=True):
                    if not api_key and arrangement_style != "Aleatorio":
                        st.error("Falta la API Key.")
                    else:
                        # Aquí es donde ocurre la magia visual del SPINNER
                        with st.spinner(f"🤖 {gemini_model} está escribiendo..."):
                            result, error = arrange_with_gemini(
                                current_word_list, 
                                arrangement_style, 
                                api_key, 
                                output_format, 
                                poet_language,
                                model_name=gemini_model,    # <--- NUEVO
                                temperature=temperature     # <--- NUEVO
                            )
                        
                        if error:
                            st.session_state.gemini_error = error
                            st.session_state.gemini_result = None
                        else:
                            st.session_state.gemini_result = result
                            st.session_state.gemini_error = None
                            
            # MOSTRAR RESULTADO
            if st.session_state.get("gemini_error"):
                st.error(st.session_state.gemini_error)
                
            if st.session_state.get("gemini_result"):
                st.success("✅ ¡Texto generado!")
                edited_result = st.text_area(
                    "Resultado (editable):",
                    value=st.session_state.gemini_result,
                    height=200,
                    key="gemini_result_editor"
                )
                # Pasar al renderizador
                direct_text = edited_result
    st.divider()
    
    # ── Font Settings ──
    st.subheader("Tipografía")

    # -- Modo Figura font (shared font slot) --
    uploaded_font = st.file_uploader(
        "Subir fuente personalizada (.ttf):",
        type=["ttf"],
        help="Opcional. Se usa Noto Sans por defecto.",
        key=f"font_uploader_{st.session_state.font_uploader_key}"
    )

    if uploaded_font is not None:
        font_bytes = uploaded_font.read()
        temp_font_path = os.path.join(tempfile.gettempdir(), "xquendart_custom_font.ttf")
        with open(temp_font_path, "wb") as f:
            f.write(font_bytes)
        st.session_state.custom_font_path = temp_font_path
        st.session_state.custom_font_bytes = font_bytes
        st.session_state.custom_font_name = uploaded_font.name
        _font_cache.clear()
        st.success("✅ Fuente personalizada cargada")

    if st.session_state.custom_font_path is not None:
        st.caption("🔤 Fuente activa: **personalizada**")
        if st.button("🔄 Restablecer a Noto Sans", use_container_width=True):
            st.session_state.custom_font_path = None
            st.session_state.custom_font_bytes = None
            st.session_state.custom_font_name = None
            st.session_state.font_uploader_key += 1
            _font_cache.clear()
            st.rerun()
    else:
        st.caption("🔤 Fuente activa: **Noto Sans** (predeterminada)")

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

    # NEW: Warning for Ultra density at low resolution
    if "density" not in dir():
        density = 80  # default, will be set properly below
    

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
                type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                help="Funciona mejor con fotos donde el sujeto es claro."
            )

            if uploaded_img is not None:
                file_key = f"{uploaded_img.name}_{uploaded_img.size}"

                # Load and cache the original image (instant — no AI processing)
                if st.session_state.get("cached_mask_key") != file_key:
                    original_image = Image.open(uploaded_img)
                    
                    MAX_DIMENSION = 2048
                    if max(original_image.size) > MAX_DIMENSION:
                        original_image.thumbnail(
                            (MAX_DIMENSION, MAX_DIMENSION), 
                            Image.Resampling.LANCZOS
                        )
                    
                    st.session_state.cached_original = original_image

                # Show original image IMMEDIATELY (before rembg runs)
                if st.session_state.get("cached_original") is not None:
                    st.image(st.session_state.cached_original, caption="Imagen original", use_container_width=True)

                # Run rembg only once per image
                if st.session_state.get("cached_mask_key") != file_key:
                    with st.spinner("🔄 Eliminando fondo (solo se hace una vez por imagen)..."):
                        try:
                            no_bg = remove_background(st.session_state.cached_original)
                            computed_mask = image_to_grayscale_mask(no_bg)
                            st.session_state.cached_mask = computed_mask
                            st.session_state.cached_mask_key = file_key
                        except Exception as e:
                            st.error(f"Error al procesar imagen: {str(e)}")
                            st.session_state.cached_mask = None
                            st.session_state.cached_mask_key = None

                # Show mask (cached or just computed)
                if st.session_state.get("cached_mask") is not None:
                    mask = st.session_state.cached_mask
                    st.image(mask, caption="Máscara generada", use_container_width=True)

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
            ["Contorno", "Silueta", "Textura"],
            key="render_mode_select"
        )
        
        # Clear previous result when render mode changes
        if "last_render_mode" not in st.session_state:
            st.session_state.last_render_mode = render_mode
        if st.session_state.last_render_mode != render_mode:
            st.session_state.generated_image = None
            st.session_state.generated_svg = None
            st.session_state.last_render_mode = render_mode

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

        # NEW: Warning for Ultra density at low resolution
        if density > 200 and output_width < 3000:
            st.info("💡 Para mejor resultado con densidad Ultra, usa resolución Grande o HD.")

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
            ["Color sólido", "Transparente", "Imagen de fondo"],
            horizontal=True
        )
        bg_transparent = bg_option == "Transparente"
        bg_color = "#FFFFFF"
        bg_image_data = None
        bg_display_mode = "Rellenar"

        if bg_option == "Color sólido":
            bg_color = st.color_picker("Color de fondo:", "#FFFFFF")
        elif bg_option == "Imagen de fondo":
            bg_image_file = st.file_uploader(
                "Sube imagen de fondo:",
                type=["png", "jpg", "jpeg", "webp", "heic", "heif"],
                key="bg_image_uploader",
                help="Esta imagen se usará como fondo del caligrama."
            )
            bg_display_mode = st.selectbox(
                "Ajuste de imagen:",
                ["Rellenar", "Ajustar", "Estirar", "Original", "Mosaico"],
                help="Rellenar: cubre todo, puede recortar. "
                     "Ajustar: toda la imagen visible, puede dejar márgenes. "
                     "Estirar: deforma para llenar. "
                     "Original: tamaño real, centrada. "
                     "Mosaico: se repite como patrón."
            )
            if bg_image_file is not None:
                bg_image_data = Image.open(bg_image_file)
                st.image(bg_image_data, caption="Vista previa del fondo", use_container_width=True)

        # ── Title & Author Signature ──
        st.markdown("---")
        with st.expander("📝 Título y firma del autor", expanded=False):

            # Shared position options (same 6 for both)
            position_labels = [
                "Arriba (centrado)",
                "Abajo (centrado)",
                "Superior izquierda",
                "Superior derecha",
                "Inferior izquierda",
                "Inferior derecha",
            ]

            title_text_input = st.text_input(
                "Título del caligrama:",
                value="",
                placeholder="Ej: Xquenda",
                help="Acepta cualquier carácter, idioma o símbolo."
            )

            col_tp, col_ts = st.columns(2)
            with col_tp:
                title_position_label = st.selectbox(
                    "Posición del título:",
                    position_labels,
                    index=0,
                    key="title_pos_select"
                )
            with col_ts:
                title_font_size = st.slider(
                    "Tamaño del título:",
                    min_value=12, max_value=120, value=48,
                    key="title_fs"
                )

            title_color = st.color_picker(
                "Color del título:",
                text_color,
                key="title_color_picker"
            )

            st.markdown("---")

            author_text_input = st.text_input(
                "Firma / Nombre del autor:",
                value="",
                placeholder="Ej: Juan Pérez - @nombredeusuario - sitioweb",
                help="Sin restricciones de caracteres."
            )

            col_ac, col_as = st.columns(2)
            with col_ac:
                author_position_label = st.selectbox(
                    "Posición de la firma:",
                    position_labels,
                    index=5,   # default: Inferior derecha
                    key="author_pos_select"
                )
            with col_as:
                author_font_size = st.slider(
                    "Tamaño de la firma:",
                    min_value=8, max_value=80, value=24,
                    key="author_fs"
                )

            author_color = st.color_picker(
                "Color de la firma:",
                text_color,
                key="author_color_picker"
            )

        # Label → internal value mapping (shared by both)
        position_map = {
            "Arriba (centrado)":    "top-center",
            "Abajo (centrado)":     "bottom-center",
            "Superior izquierda":   "top-left",
            "Superior derecha":     "top-right",
            "Inferior izquierda":   "bottom-left",
            "Inferior derecha":     "bottom-right",
        }

        
    # ── Generate ──
    st.divider()

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
                "custom_font_path": st.session_state.custom_font_path,
                "bg_image": bg_image_data,
                "bg_display_mode": bg_display_mode,
            }

            with st.spinner("🎨 Generando caligrama..."):
                result_image, svg_string = fill_shape_with_text(mask, render_text, config)

                # ── Add title and author signature on top ──
                result_image, svg_string = add_title_and_author(
                    result_image, svg_string,
                    title_text=title_text_input,
                    author_text=author_text_input,
                    title_font_size=title_font_size,
                    author_font_size=author_font_size,
                    title_position=position_map.get(title_position_label, "top-center"),
                    author_position=position_map.get(author_position_label, "bottom-right"),
                    title_color=title_color,
                    author_color=author_color,
                    custom_font_path=st.session_state.custom_font_path,
                )

                st.session_state.generated_image = result_image
                st.session_state.generated_svg = svg_string
                st.session_state.last_render_mode = render_mode

            st.success("✅ ¡Caligrama generado!")

        # Display result from session state (persists across reruns)
        if st.session_state.generated_image is not None:
            st.image(st.session_state.generated_image, caption="Tu caligrama", use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                buf = io.BytesIO()
                st.session_state.generated_image.save(buf, format="PNG")
                st.download_button(
                    "📥 Descargar PNG",
                    data=buf.getvalue(),
                    file_name="xquendart_caligrama.png",
                    mime="image/png",
                    use_container_width=True
                )
            with col_dl2:
                if st.session_state.generated_svg:
                    st.download_button(
                        "📥 Descargar SVG",
                        data=st.session_state.generated_svg,
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

    render_text = get_text_for_rendering(
        st.session_state.parsed_lists,
        st.session_state.parsed_poems,
        st.session_state.text_source,
        direct_text=direct_text,
        selected_list=selected_list_name,
        selected_poem_idx=selected_poem_idx,
        preserve_edge_spaces=True,
    )
    text_sources = make_lienzo_text_sources(
        st.session_state.parsed_lists,
        st.session_state.parsed_poems,
        render_text
    )

    if not text_sources:
        st.info("👈 Escribe, carga o genera texto en la barra lateral para comenzar.")
    else:
        lienzo_config = {
            "width": int(output_width),
            "height": int(output_height),
            "textSources": text_sources,
            "font": get_lienzo_font_payload(render_text),
        }
        aspect = output_height / max(output_width, 1)
        component_height = min(1120, max(720, int(820 * aspect) + 260))
        components.html(
            build_lienzo_canvas_html(lienzo_config),
            height=component_height,
            scrolling=True
        )

# ═══════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════

st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "XquendArt — Creado para los poetas indígenas de México<br>"
    "Xquenda_Lab · 2026"
    "</div>",
    unsafe_allow_html=True
)
