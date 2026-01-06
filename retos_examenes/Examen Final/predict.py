import tensorflow as tf
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def scanner_preprocess(image_path):
    """
    Preprocesamiento tipo escáner profesional (versión ajustada).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Corrección de iluminación
    kernel_size = 31
    background = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img_corrected = cv2.divide(img, background, scale=255)
    
    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_corrected)
    
    # 3. Denoise MUY SUAVE (preservar detalles)
    img_denoised = cv2.fastNlMeansDenoising(img_clahe, None, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # 4. Binarización con Otsu
    _, binary = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 5. Limpieza morfológica MÁS SUAVE
    kernel_open = np.ones((2, 2), np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 6. Eliminar componentes pequeños PERO MÁS PERMISIVO
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if len(areas) > 0:
        img_height, img_width = img.shape
        
        min_area_absolute = (img_height * img_width) * 0.0001
        max_area = np.max(areas)
        min_area_threshold = max(min_area_absolute, max_area * 0.01)
        
        clean_binary = np.zeros_like(binary_clean)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area_threshold:
                clean_binary[labels == i] = 255
        
        if np.sum(clean_binary) < 100:
            print("⚠️ Limpieza demasiado agresiva, devolviendo binarización sin limpieza")
            return binary_clean
        
        return clean_binary
    else:
        return binary_clean

def preprocess_image(img):
    """
    Aplica preprocesamiento a una imagen para mejorar la segmentación y predicción.
    """
    # Suavizado
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Ajuste de contraste
    img = cv2.equalizeHist(img)
    return img

def segment_image(image_path):
    """
    Segmenta caracteres usando preprocesamiento tipo escáner.
    """
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # USAR PREPROCESAMIENTO TIPO ESCÁNER
    thresh = scanner_preprocess(image_path)
    
    print("✅ Preprocesamiento tipo escáner completado")
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"🔍 Contornos detectados: {len(contours)}")
    
    if len(contours) == 0:
        print("⚠️ No se detectaron contornos")
        return np.array([]), []
    
    # Estadísticas de imagen
    img_height, img_width = thresh.shape
    
    # Recolectar candidatos
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = h / w if w > 0 else 0
        
        candidates.append({
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio
        })
    
    if not candidates:
        return np.array([]), []
    
    # Calcular estadísticas
    areas = [c['area'] for c in candidates]
    heights = [c['bbox'][3] for c in candidates]
    
    median_area = np.median(areas)
    median_height = np.median(heights)
    
    # Filtros MÁS PERMISIVOS
    min_area = max(50, median_area * 0.15)  # ← Reducido de 0.3 a 0.15
    max_area = median_area * 10  # ← Aumentado de 4 a 10
    
    min_height = max(10, int(median_height * 0.4))  # ← Más permisivo
    max_height = int(img_height * 0.98)
    min_width = 3  # ← Reducido
    max_width = int(img_width * 0.5)  # ← Aumentado
    
    # Filtrar
    valid_candidates = []
    for c in candidates:
        x, y, w, h = c['bbox']
        area = c['area']
        aspect_ratio = c['aspect_ratio']
        
        # Filtros más permisivos
        if (min_area < area < max_area and
            min_width < w < max_width and
            min_height < h < max_height and
            0.2 < aspect_ratio < 6):  # ← Rango más amplio
            
            valid_candidates.append(c)
    
    print(f"✅ Caracteres válidos: {len(valid_candidates)}")
    
    if not valid_candidates:
        print("⚠️ No se encontraron caracteres válidos después del filtrado")
        
        # FALLBACK: Si no hay caracteres válidos, ser AÚN MÁS PERMISIVO
        print("🔄 Intentando con filtros mínimos...")
        valid_candidates = []
        
        for c in candidates:
            x, y, w, h = c['bbox']
            
            # Solo filtros básicos
            if w > 5 and h > 10:
                valid_candidates.append(c)
        
        print(f"   Caracteres con filtros mínimos: {len(valid_candidates)}")
        
        if not valid_candidates:
            return np.array([]), []
    
    # Extraer caracteres
    char_images = []
    bounding_boxes = []
    
    for c in valid_candidates:
        x, y, w, h = c['bbox']
        
        char_img = original[y:y + h, x:x + w]
        char_img = add_padding(char_img, target_size=32)
        char_img = char_img / 255.0
        
        char_images.append(char_img)
        bounding_boxes.append((x, y, w, h))
    
    # Ordenar
    if char_images:
        sorted_data = sorted(zip(bounding_boxes, char_images), key=lambda b: b[0][0])
        bounding_boxes, char_images = zip(*sorted_data)
        
        return np.array(char_images).reshape(-1, 32, 32, 1), bounding_boxes
    else:
        return np.array([]), []

def add_padding(img, target_size=32):
    """
    Añade padding a una imagen para que tenga el tamaño objetivo sin distorsión.
    """
    h, w = img.shape
    
    # Calcular el factor de escala manteniendo el aspect ratio
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Redimensionar
    resized = cv2.resize(img, (new_w, new_h))
    
    # Crear imagen con padding
    padded = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Centrar la imagen redimensionada
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded

def detect_spaces(bounding_boxes, avg_char_width_multiplier=2.0):
    """
    Detecta espacios entre palabras.
    Más permisivo para escritura manuscrita.
    """
    if len(bounding_boxes) < 2:
        return []
    
    # Calcular distancias entre caracteres consecutivos
    gaps = []
    for i in range(len(bounding_boxes) - 1):
        x1_end = bounding_boxes[i][0] + bounding_boxes[i][2]
        x2_start = bounding_boxes[i + 1][0]
        gap = x2_start - x1_end
        gaps.append(gap)
    
    if not gaps:
        return []
    
    # Usar la mediana de gaps como referencia
    median_gap = np.median(gaps)
    
    space_positions = []
    for i, gap in enumerate(gaps):
        # Si el gap es significativamente mayor que la mediana, es un espacio
        if gap > median_gap * avg_char_width_multiplier:
            space_positions.append(i + 1)
    
    return space_positions

def predict_image(image_path, prediction_type, model=None):
    """Realiza predicciones para imágenes (frases o caracteres)."""
    if model is None:
        model = tf.keras.models.load_model("ocr_model.h5")  # Cargar modelo si no se proporciona

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if prediction_type == "phrase":
        # Segmentar la imagen en caracteres individuales
        char_images, bounding_boxes = segment_image(image_path)
        if char_images.size == 0:
            print("No se detectaron caracteres en la imagen.")
            return ""

        predictions = model.predict(char_images)
        predicted_classes = np.argmax(predictions, axis=1)

        # Detectar espacios entre palabras
        space_positions = detect_spaces(bounding_boxes)

        # Decodificar predicciones
        predicted_chars = []
        for idx, cls in enumerate(predicted_classes):
            # Agregar espacio si corresponde (antes del carácter actual)
            if idx in space_positions:
                predicted_chars.append(" ")
            
            # Decodificar el carácter
            if 0 <= cls <= 9:
                predicted_chars.append(chr(cls + ord('0')))
            elif 10 <= cls <= 35:
                predicted_chars.append(chr(cls - 10 + ord('A')))
            elif 36 <= cls <= 61:
                predicted_chars.append(chr(cls - 36 + ord('a')))
            else:
                predicted_chars.append("?")

        predicted_phrase = "".join(predicted_chars)

        # Visualizar predicción
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Dibujar rectángulos y etiquetas para cada carácter
        for (x, y, w, h), char in zip(bounding_boxes, predicted_chars):
            if char != " ":  # No dibujar rectángulos para espacios
                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_color, char, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        plt.figure(figsize=(15, 6))
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.title(f"Predicción (Frase): {predicted_phrase}", fontsize=16, fontweight='bold')
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        print(f"Predicción (Frase): {predicted_phrase}")
        print(f"Total de caracteres detectados: {len(predicted_chars)}")
        return predicted_phrase
        
    else:
        # Predicción individual de letra o número
        img_resized = cv2.resize(img, (32, 32)) / 255.0
        img_resized = img_resized.reshape(1, 32, 32, 1)
        prediction = model.predict(img_resized)
        predicted_class = np.argmax(prediction)

        if prediction_type == "letter":
            if 0 <= predicted_class <= 9:
                predicted_char = chr(predicted_class + ord('0'))
            elif 10 <= predicted_class <= 35:
                predicted_char = chr(predicted_class - 10 + ord('A'))
            elif 36 <= predicted_class <= 61:
                predicted_char = chr(predicted_class - 36 + ord('a'))
            else:
                predicted_char = "Desconocido"
        elif prediction_type == "number":
            if 0 <= predicted_class <= 9:
                predicted_char = chr(predicted_class + ord('0'))
            else:
                predicted_char = "No es un número"
        else:
            predicted_char = "Desconocido"

        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="gray")
        plt.title(f"Predicción: {predicted_char}", fontsize=14, fontweight='bold')
        plt.axis("off")
        plt.show()

        print(f"Predicción: Clase {predicted_class}, Carácter: {predicted_char}")
        return predicted_char

def predict_folder(folder_path, model):
    predictions = []
    true_labels = []
    correct_predictions = 0
    total_predictions = 0
    total_images = 0

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        print(f"Procesando: {img_name}")

        if "label_" in img_name:
            label = img_name.split("label_")[1].split(".")[0]
            prediction_type = "letter" if len(label) == 1 and label.isalnum() else "number"
        else:
            # Usar la primera letra del nombre como etiqueta si es alfanumérica
            potential_label = img_name[0]  # Primera letra del nombre
            if potential_label.isdigit():
                prediction_type = "number"
                label = potential_label
            elif potential_label.isalpha():
                prediction_type = "letter"
                label = potential_label
            else:
                print(f"No se pudo determinar la etiqueta para {img_name}.")
                continue
            
        try:
            prediction = predict_image(img_path, prediction_type, model)  # Usar predict_image
            predictions.append(prediction)

            # Comparar con la etiqueta verdadera (si existe)
            if "label_" in img_name:
                true_label = img_name.split("label_")[1].split(".")[0]
                true_labels.append(true_label)
                if prediction == true_label:
                    correct_predictions += 1
                total_predictions += 1
        except Exception as e:
            print(f"Error procesando {img_name}: {e}")

    # Calcular precisión si hay etiquetas
    # Calcular precisión
    if total_images > 0:
        accuracy = (correct_predictions / total_images) * 100
        print(f"Precisión calculada: {accuracy:.2f}%")
        return accuracy
    else:
        print("No se procesaron imágenes válidas.")
        return 0

def debug_segmentation(image_path):
    """
    Visualiza el preprocesamiento tipo escáner paso a paso.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Paso 1: Corrección de iluminación
    kernel_size = 51
    background = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img_corrected = cv2.divide(img, background, scale=255)
    
    # Paso 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_corrected)
    
    # Paso 3: Denoise
    img_denoised = cv2.fastNlMeansDenoising(img_clahe, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Paso 4: Binarización Otsu
    _, binary = cv2.threshold(img_denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Paso 5: Limpieza morfológica
    kernel = np.ones((2, 2), np.uint8)
    binary_morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Paso 6: Eliminar ruido (componentes pequeños)
    clean_binary = scanner_preprocess(image_path)
    
    # Paso 7: Contornos
    contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img_height, img_width = img.shape
    min_area = 100
    
    valid_count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        if area > min_area and h > img_height * 0.3:
            cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 255, 0), 3)
            valid_count += 1
        else:
            cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 0, 255), 1)
    
    # Visualizar
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("1. Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(background, cmap='gray')
    axes[0, 1].set_title("2. Fondo estimado", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_corrected, cmap='gray')
    axes[0, 2].set_title("3. Iluminación corregida", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(img_clahe, cmap='gray')
    axes[1, 0].set_title("4. CLAHE (contraste adaptativo)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_denoised, cmap='gray')
    axes[1, 1].set_title("5. Denoising (reducción ruido)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(binary, cmap='gray')
    axes[1, 2].set_title("6. Binarización Otsu", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(binary_morph, cmap='gray')
    axes[2, 0].set_title("7. Morfología (close)", fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(clean_binary, cmap='gray')
    axes[2, 1].set_title("8. ✨ LIMPIO (componentes pequeños eliminados)", fontsize=12, fontweight='bold', color='green')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    axes[2, 2].set_title(f"9. Contornos: {len(contours)} total, {valid_count} válidos", fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 ESTADÍSTICAS FINALES:")
    print(f"   Contornos totales: {len(contours)}")
    print(f"   Contornos válidos (verde): {valid_count}")
    print(f"   Ruido eliminado (rojo): {len(contours) - valid_count}")
