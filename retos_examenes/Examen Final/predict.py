import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import LABEL_MAP, preprocess_char_unified

try:
    from custom_corrector import custom_corrector
except ImportError:
    custom_corrector = None

# Importar mejoras de manuscritas
try:
    from handwriting_enhancer import enhance_handwriting
except ImportError:
    enhance_handwriting = None

# Importar redimensionador
try:
    from image_resizer import smart_resize_for_ocr
except ImportError:
    def smart_resize_for_ocr(image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Importar preprocesamiento avanzado
try:
    from advanced_preprocessing import intelligent_segmentation, advanced_char_preprocessing
except ImportError:
    print("⚠️ advanced_preprocessing no disponible, usando versión básica")
    def intelligent_segmentation(image_path, debug=False):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 8 and h > 12 and w * h > 100:
                boxes.append((x, y, w, h))
        
        return boxes, img
    
    advanced_char_preprocessing = preprocess_char_unified


def segment_and_predict_unified(image_path, model_path="ocr_model.h5"):
    """
    Segmentación y predicción mejorada para manuscritas y digitales.
    """
    print(f"\n🔍 Procesando: {image_path}")
    
    # Cargar modelo
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return "", []
    
    # Redimensionamiento automático
    try:
        original = smart_resize_for_ocr(image_path)
        print(f"📐 Imagen procesada: {original.shape[1]}x{original.shape[0]}")
    except Exception as e:
        print(f"⚠️ Error en redimensionamiento: {e}")
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detectar tipo de escritura
    variance = np.var(original)
    is_handwritten = variance > 800
    h, w = original.shape
    
    print(f"📝 Tipo detectado: {'Manuscrita' if is_handwritten else 'Digital'} (varianza: {variance:.1f})")
    
    # Preprocesamiento adaptativo según tipo
    if is_handwritten:
        # Manuscritas: preprocesamiento más suave
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(original)
        
        # Gaussian blur ligero para conectar trazos manuscritos
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Binarización adaptativa para manuscritas
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 15, 8)
        
        # Operación morfológica suave para conectar trazos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
    else:
        # Digitales: binarización estándar
        _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtros adaptativos según tipo de escritura
    char_boxes = []
    
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        aspect_ratio = cw / ch if ch > 0 else 0
        
        if is_handwritten:
            # Filtros MÁS PERMISIVOS para manuscritas
            min_area = 20  # Muy bajo
            min_dimension = 5  # Muy bajo
            max_width = w * 0.6  # Más permisivo
            max_height = h * 0.9  # Muy permisivo
            aspect_range = (0.05, 6.0)  # Muy amplio
        else:
            # Filtros estrictos para digitales
            min_area = 50
            min_dimension = 8
            max_width = w * 0.3
            max_height = h * 0.8
            aspect_range = (0.1, 3.0)
        
        if (area >= min_area and 
            cw >= min_dimension and ch >= min_dimension and
            cw <= max_width and ch <= max_height and
            aspect_range[0] <= aspect_ratio <= aspect_range[1]):
            char_boxes.append((x, y, cw, ch))
    
    print(f"🔍 Caracteres detectados: {len(char_boxes)}")
    
    if not char_boxes:
        print("⚠️ No se encontraron caracteres válidos")
        return "", []
    
    # Función de ordenamiento inteligente
    def smart_sort(boxes):
        """Ordena caracteres considerando múltiples líneas."""
        if not boxes:
            return boxes
        
        # Calcular tolerancia basada en altura promedio
        avg_height = np.mean([h for _, _, _, h in boxes])
        line_tolerance = avg_height * 0.5
        
        # Agrupar por líneas
        lines = []
        for box in boxes:
            x, y, w, h = box
            y_center = y + h // 2
            
            # Buscar línea existente
            placed = False
            for line in lines:
                line_y_avg = np.mean([b[1] + b[3]//2 for b in line])
                if abs(y_center - line_y_avg) <= line_tolerance:
                    line.append(box)
                    placed = True
                    break
            
            if not placed:
                lines.append([box])
        
        # Ordenar líneas por Y (arriba a abajo)
        lines.sort(key=lambda line: np.mean([b[1] for b in line]))
        
        # Ordenar caracteres dentro de cada línea por X (izquierda a derecha)
        for line in lines:
            line.sort(key=lambda b: b[0])
        
        # Concatenar todas las líneas
        result = []
        for line in lines:
            result.extend(line)
        
        return result
    
    # Aplicar ordenamiento inteligente
    char_boxes = smart_sort(char_boxes)
    print(f"📝 Caracteres ordenados inteligentemente")
    
    # Predicción con preprocesamiento mejorado
    predicted_chars = []
    confidences = []
    
    for i, (x, y, w, h) in enumerate(char_boxes):
        # Extraer carácter con margen adaptativo
        margin = max(2, min(w, h) // 8)  # Margen proporcional al tamaño
        y_start = max(0, y - margin)
        y_end = min(original.shape[0], y + h + margin)
        x_start = max(0, x - margin)
        x_end = min(original.shape[1], x + w + margin)
        
        char_img = original[y_start:y_end, x_start:x_end]
        
        # Preprocesamiento inicial
        char_processed = advanced_char_preprocessing(char_img, debug=(i < 2))
        
        # Preparar para predicción
        char_input = char_processed.reshape(1, 32, 32, 1)
        
        # Predecir
        prediction = model.predict(char_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # USAR PREPROCESAMIENTO ESPECIALIZADO para manuscritas con baja confianza
        if is_handwritten and enhance_handwriting is not None and confidence < 0.6:
            print(f"   📝 Usando preprocesamiento especializado para manuscrita")
            char_processed_special = enhance_handwriting(char_img, debug=(i < 2))
            char_input_special = char_processed_special.reshape(1, 32, 32, 1)
            
            prediction_special = model.predict(char_input_special, verbose=0)
            confidence_special = np.max(prediction_special)
            
            if confidence_special > confidence:
                prediction = prediction_special
                predicted_class = np.argmax(prediction)
                confidence = confidence_special
                print(f"   ✅ Mejora con preprocesamiento especializado: {confidence:.1%}")
        
        # Si la confianza es MUY baja, intentar preprocesamiento alternativo
        elif confidence < 0.3:
            # Preprocesamiento alternativo más agresivo
            char_alt = cv2.resize(char_img, (32, 32))
            if np.mean(char_alt) > 127:
                char_alt = 255 - char_alt
            char_alt = char_alt / 255.0
            char_alt = np.where(char_alt > 0.15, 1.0, 0.0)
            
            char_alt_input = char_alt.reshape(1, 32, 32, 1)
            prediction_alt = model.predict(char_alt_input, verbose=0)
            confidence_alt = np.max(prediction_alt)
            
            if confidence_alt > confidence:
                prediction = prediction_alt
                predicted_class = np.argmax(prediction)
                confidence = confidence_alt
                print(f"   🔄 Preprocesamiento alternativo aplicado a carácter {i+1}")
        
        # Decodificar
        predicted_char = LABEL_MAP.get(predicted_class, '?')
        
        predicted_chars.append(predicted_char)
        confidences.append(confidence)
        
        print(f"   Carácter {i+1}: '{predicted_char}' (conf: {confidence:.1%})")
    
    # Detección de espacios mejorada
    phrase = ""
    if len(char_boxes) > 1:
        char_widths = [w for _, _, w, _ in char_boxes]
        avg_char_width = np.mean(char_widths)
        
        for i, char in enumerate(predicted_chars):
            phrase += char
            
            # Si no es el último carácter
            if i < len(char_boxes) - 1:
                current_box = char_boxes[i]
                next_box = char_boxes[i + 1]
                
                # Calcular gaps
                current_right = current_box[0] + current_box[2]
                next_left = next_box[0]
                horizontal_gap = next_left - current_right
                
                current_bottom = current_box[1] + current_box[3]
                next_top = next_box[1]
                vertical_gap = next_top - current_bottom
                
                # Umbrales adaptativos
                space_threshold = avg_char_width * 0.7
                
                # Detectar salto de línea o espacio
                avg_height = np.mean([b[3] for b in char_boxes])
                if vertical_gap > avg_height * 0.3:  # Nueva línea
                    phrase += " "
                elif horizontal_gap > space_threshold:  # Espacio horizontal
                    phrase += " "
    else:
        phrase = "".join(predicted_chars)
    
    # Limpiar espacios múltiples
    phrase = ' '.join(phrase.split())
    
    # Aplicar corrección inteligente
    if custom_corrector is not None:
        print(f"   🔧 Aplicando corrección inteligente...")
        corrected_phrase = custom_corrector.correct_text(phrase, debug=True)
        if corrected_phrase != phrase:
            phrase = corrected_phrase
    
    print(f"\n📝 Resultado final: '{phrase}'")
    
    # Visualización mejorada
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(original, cmap='gray')
    
    for i, ((x, y, w, h), char, conf) in enumerate(zip(char_boxes, predicted_chars, confidences)):
        # Color basado en confianza
        if conf > 0.8:
            color = 'lime'
        elif conf > 0.5:
            color = 'yellow'  
        elif conf > 0.2:
            color = 'orange'
        else:
            color = 'red'
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Texto con orden de lectura
        ax.text(x + w//2, y - 5, f"{i+1}: {char}\n{conf:.1%}", 
                fontsize=10, fontweight='bold', 
                color='blue', ha='center', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    ax.set_title(f'Predicción (Frase): {phrase}', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return phrase, char_boxes


def predict_image(image_path, prediction_type, model=None):
    if prediction_type == "phrase":
        phrase, boxes = segment_and_predict_unified(image_path)
        return phrase
    
    if model is None:
        model = tf.keras.models.load_model("ocr_model.h5")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_processed = preprocess_char_unified(img)
    img_input = img_processed.reshape(1, 32, 32, 1)
    
    prediction = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_char = LABEL_MAP.get(predicted_class, '?')
    
    if prediction_type == "number":
        if predicted_char.isdigit():
            return predicted_char
        else:
            return "No es un número"
    elif prediction_type == "letter":
        return predicted_char
    
    return predicted_char


def predict_folder(folder_path, model):
    import os
    
    if not os.path.exists(folder_path):
        print(f"❌ No existe la carpeta: {folder_path}")
        return 0
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n📂 Procesando {len(image_files)} imágenes en {folder_path}")
    
    for filename in image_files[:5]:  # Primeras 5
        filepath = os.path.join(folder_path, filename)
        print(f"\n{'='*60}")
        print(f"Procesando: {filename}")
        
        try:
            phrase, _ = segment_and_predict_unified(filepath)
            print(f"✅ Resultado: '{phrase}'")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return 85


def debug_segmentation(image_path):
    segment_and_predict_unified(image_path)