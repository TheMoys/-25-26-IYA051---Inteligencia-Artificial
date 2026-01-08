import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import LABEL_MAP, preprocess_char_unified


def enhanced_preprocess(image_path):
    """
    Preprocesamiento avanzado para fotos manuscritas.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Corrección de iluminación
    kernel_size = min(99, max(img.shape) // 4)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    background = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    corrected = cv2.divide(img, background, scale=255)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=8, templateWindowSize=7, searchWindowSize=21)
    
    # Binarización
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return binary


def segment_and_predict_unified(image_path, model_path="ocr_model.h5"):
    """
    Segmenta y predice usando el MISMO preprocesamiento que el entrenamiento.
    """
    print(f"\n🔍 Procesando: {image_path}")
    
    # Cargar modelo
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return "", []
    
    # Cargar imagen
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("❌ No se pudo cargar la imagen")
        return "", []
    
    print(f"📊 Imagen original: {original.shape}")
    
    # Detectar tipo de imagen
    variance = np.var(original)
    mean_intensity = np.mean(original)
    
    print(f"   Varianza: {variance:.1f}, Media: {mean_intensity:.1f}")
    
    if variance > 800:  # Imagen con mucho ruido (foto)
        print("   📷 Tipo: Foto manuscrita")
        binary = enhanced_preprocess(image_path)
    else:  # Imagen digital limpia
        print("   📄 Tipo: Imagen digital")
        _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"🔍 Contornos detectados: {len(contours)}")
    
    # Filtrar contornos
    img_height, img_width = original.shape
    
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # Filtros básicos
        if (w >= 8 and h >= 12 and 
            w <= img_width * 0.5 and h <= img_height * 0.8 and
            area >= 100):
            valid_contours.append((x, y, w, h))
    
    print(f"✅ Caracteres válidos: {len(valid_contours)}")
    
    if not valid_contours:
        print("⚠️ No se encontraron caracteres válidos")
        return "", []
    
    # Ordenar por posición horizontal
    valid_contours = sorted(valid_contours, key=lambda b: b[0])
    
    # Predicción con preprocesamiento unificado
    predicted_chars = []
    char_boxes = []
    
    for i, (x, y, w, h) in enumerate(valid_contours):
        # Extraer carácter del ORIGINAL
        char_img = original[y:y+h, x:x+w]
        
        # USAR LA MISMA FUNCIÓN DE PREPROCESAMIENTO QUE EN ENTRENAMIENTO
        char_processed = preprocess_char_unified(char_img)
        
        # Preparar para predicción
        char_input = char_processed.reshape(1, 32, 32, 1)
        
        # Predecir
        prediction = model.predict(char_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Decodificar
        predicted_char = LABEL_MAP.get(predicted_class, '?')
        
        predicted_chars.append(predicted_char)
        char_boxes.append((x, y, w, h))
        
        print(f"   Carácter {i+1}: '{predicted_char}' (confianza: {confidence:.2%})")
    
    # Detectar espacios y formar frase
    if len(char_boxes) > 1:
        gaps = []
        for i in range(len(char_boxes) - 1):
            current_right = char_boxes[i][0] + char_boxes[i][2]
            next_left = char_boxes[i + 1][0]
            gap = next_left - current_right
            gaps.append(gap)
        
        if gaps:
            avg_gap = np.mean(gaps)
            space_threshold = avg_gap * 2.0
            
            phrase = ""
            for i, char in enumerate(predicted_chars):
                phrase += char
                if i < len(gaps) and gaps[i] > space_threshold:
                    phrase += " "
        else:
            phrase = "".join(predicted_chars)
    else:
        phrase = "".join(predicted_chars)
    
    print(f"\n📝 Resultado: '{phrase}'")
    
    # Visualización
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.imshow(original, cmap='gray')
    
    for i, ((x, y, w, h), char) in enumerate(zip(char_boxes, predicted_chars)):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        
        ax.text(x, y-5, char, fontsize=14, fontweight='bold', color='blue',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title(f'Predicción (Frase): {phrase}', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return phrase, char_boxes


def predict_image(image_path, prediction_type, model=None):
    """
    Función legacy para compatibilidad.
    """
    if prediction_type == "phrase":
        phrase, boxes = segment_and_predict_unified(image_path)
        return phrase
    
    # Para predicción individual
    if model is None:
        model = tf.keras.models.load_model("ocr_model.h5")
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_processed = preprocess_char_unified(img)
    img_input = img_processed.reshape(1, 32, 32, 1)
    
    prediction = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_char = LABEL_MAP.get(predicted_class, '?')
    
    # Filtrar según tipo
    if prediction_type == "number":
        if predicted_char.isdigit():
            return predicted_char
        else:
            return "No es un número"
    elif prediction_type == "letter":
        return predicted_char
    
    return predicted_char


def predict_folder(folder_path, model):
    """
    Predice todas las imágenes en una carpeta.
    """
    import os
    
    if not os.path.exists(folder_path):
        print(f"❌ No existe la carpeta: {folder_path}")
        return 0
    
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\n📂 Procesando {len(image_files)} imágenes en {folder_path}")
    
    correct = 0
    total = 0
    
    for filename in image_files[:10]:  # Primeras 10
        filepath = os.path.join(folder_path, filename)
        print(f"\n{'='*60}")
        print(f"Procesando: {filename}")
        
        try:
            phrase, _ = segment_and_predict_unified(filepath)
            print(f"✅ Resultado: '{phrase}'")
            
            # Extraer etiqueta esperada
            expected_label = extract_label_from_filename(filename)
            if expected_label and phrase.strip():
                if phrase[0] == expected_label:
                    correct += 1
                total += 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n📊 Accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy
    
    return 0


def debug_segmentation(image_path):
    """
    Función legacy para compatibilidad.
    """
    segment_and_predict_unified(image_path)


def extract_label_from_filename(filename):
    """
    Extrae la etiqueta esperada del nombre del archivo.
    """
    try:
        if "label_" in filename:
            label_num = int(filename.split("label_")[1].split('.')[0])
            if 0 <= label_num <= 9:
                return str(label_num)
            elif 10 <= label_num <= 35:
                return chr(label_num - 10 + ord('A'))
            elif 36 <= label_num <= 61:
                return chr(label_num - 36 + ord('a'))
        elif len(filename) > 1 and filename[1] == '_':
            return filename[0]
    except:
        pass
    return None


# Funciones legacy para compatibilidad
def scanner_preprocess(image_path):
    return enhanced_preprocess(image_path)

def preprocess_image(img):
    return img

def segment_image(image_path):
    char_images, boxes = segment_and_predict_unified(image_path)
    return char_images, boxes

def add_padding(img, target_size=32):
    return cv2.resize(img, (target_size, target_size))

def detect_spaces(bounding_boxes, avg_char_width_multiplier=2.0):
    return []