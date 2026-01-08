import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import LABEL_MAP, preprocess_char_unified

# Importar nuevo preprocesamiento
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
    Segmentación y predicción mejorada.
    """
    print(f"\n🔍 Procesando: {image_path}")
    
    # Cargar modelo
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Modelo cargado")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return "", []
    
    # USAR SEGMENTACIÓN INTELIGENTE
    try:
        char_boxes, original = intelligent_segmentation(image_path, debug=True)
        print(f"� Segmentación inteligente: {len(char_boxes)} caracteres")
    except Exception as e:
        print(f"⚠️ Error en segmentación inteligente: {e}")
        # Fallback a segmentación básica
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_boxes = [(cv2.boundingRect(c)) for c in contours if cv2.boundingRect(c)[2] > 8]
    
    if not char_boxes:
        print("⚠️ No se encontraron caracteres")
        return "", []
    
    # Ordenar por posición horizontal
    char_boxes = sorted(char_boxes, key=lambda b: b[0])
    
    # Predicción con preprocesamiento mejorado
    predicted_chars = []
    confidences = []
    
    for i, (x, y, w, h) in enumerate(char_boxes):
        # Extraer carácter con margen
        margin = 2
        y_start = max(0, y - margin)
        y_end = min(original.shape[0], y + h + margin)
        x_start = max(0, x - margin)
        x_end = min(original.shape[1], x + w + margin)
        
        char_img = original[y_start:y_end, x_start:x_end]
        
        # USAR PREPROCESAMIENTO AVANZADO
        char_processed = advanced_char_preprocessing(char_img, debug=(i < 3))  # Debug primeros 3
        
        # Preparar para predicción
        char_input = char_processed.reshape(1, 32, 32, 1)
        
        # Predecir
        prediction = model.predict(char_input, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Decodificar
        predicted_char = LABEL_MAP.get(predicted_class, '?')
        
        predicted_chars.append(predicted_char)
        confidences.append(confidence)
        
        print(f"   Carácter {i+1}: '{predicted_char}' (conf: {confidence:.2%})")
    
    # Detección de espacios mejorada
    if len(char_boxes) > 1:
        gaps = []
        char_widths = [w for _, _, w, _ in char_boxes]
        avg_char_width = np.mean(char_widths)
        
        for i in range(len(char_boxes) - 1):
            current_right = char_boxes[i][0] + char_boxes[i][2]
            next_left = char_boxes[i + 1][0]
            gap = next_left - current_right
            gaps.append(gap)
        
        # Umbral dinámico basado en ancho promedio de caracteres
        space_threshold = avg_char_width * 0.8
        
        phrase = ""
        for i, char in enumerate(predicted_chars):
            phrase += char
            if i < len(gaps) and gaps[i] > space_threshold:
                phrase += " "
    else:
        phrase = "".join(predicted_chars)
    
    print(f"\n📝 Resultado: '{phrase}'")
    
    # Visualización mejorada
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(original, cmap='gray')
    
    for i, ((x, y, w, h), char, conf) in enumerate(zip(char_boxes, predicted_chars, confidences)):
        # Color basado en confianza
        color = 'lime' if conf > 0.8 else 'yellow' if conf > 0.6 else 'red'
        
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Texto con confianza
        ax.text(x, y-5, f"{char}\n{conf:.1%}", fontsize=12, fontweight='bold', 
                color='blue', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    ax.set_title(f'Predicción (Frase): {phrase}', fontsize=18, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return phrase, char_boxes

# Resto de funciones iguales...
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

# Resto de funciones legacy...
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
    
    return 85  # Valor dummy

def debug_segmentation(image_path):
    segment_and_predict_unified(image_path)