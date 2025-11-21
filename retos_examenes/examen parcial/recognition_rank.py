import cv2
import numpy as np
import os

def load_rank_templates(path='templates/ranks', size=(70, 50)):  # ORDEN CORRECTO: alto x ancho
    """Carga las plantillas de los números/letras"""
    templates = {}
    
    if not os.path.exists(path):
        print(f"[ERROR] No existe el directorio: {path}")
        return templates
    
    for name in os.listdir(path):
        if not name.lower().endswith('.png'):
            continue
        
        filepath = os.path.join(path, name)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"[WARNING] No se pudo cargar: {filepath}")
            continue
        
        # Las plantillas son NEGRAS sobre BLANCO -> convertir a BLANCO sobre NEGRO
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # INV para invertir
        
        # Redimensionar (mantener aspect ratio)
        resized = cv2.resize(binary, (size[1], size[0]), interpolation=cv2.INTER_AREA)  # (width, height)
        
        # Limpiar ruido
        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        rank_name = os.path.splitext(name)[0]
        templates[rank_name] = clean
        
        print(f"[LOAD] Template '{rank_name}' cargado: {clean.shape}")
    
    print(f"\n[INIT] Total plantillas cargadas: {len(templates)}")
    return templates

# Cargar plantillas globales
RANK_TEMPLATES = load_rank_templates()

def preprocess_rank_roi(roi):
    """
    Preprocesa el ROI del rank: BLANCO sobre NEGRO
    """
    # Convertir a grises
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Ecualizar histograma para mejorar contraste
    gray = cv2.equalizeHist(gray)
    
    # Threshold de Otsu (más robusto)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Si hay más fondo que símbolo, invertir
    white_pixels = cv2.countNonZero(binary)
    if white_pixels < binary.size * 0.2:  # Muy poco blanco = necesita inversión
        binary = cv2.bitwise_not(binary)
    
    # Limpieza agresiva
    kernel = np.ones((3, 3), np.uint8)
    
    # Eliminar ruido pequeño
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Cerrar huecos
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Encontrar el contorno principal y recortar
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Obtener el contorno más grande
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Añadir padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary.shape[1] - x, w + 2*padding)
        h = min(binary.shape[0] - y, h + 2*padding)
        
        # Recortar
        cropped = binary[y:y+h, x:x+w]
        
        # Crear imagen con padding para mantener aspect ratio
        max_dim = max(h, w)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        # Centrar el símbolo
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
        
        return padded
    
    return binary

def match_template_multimethod(img, template):
    """
    Aplica múltiples métodos de template matching
    """
    # Asegurarse de que ambas imágenes tienen el mismo tamaño
    if img.shape != template.shape:
        img = cv2.resize(img, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_AREA)
    
    methods = [
        (cv2.TM_CCOEFF_NORMED, 0.4),    # 40% peso
        (cv2.TM_CCORR_NORMED, 0.3),     # 30% peso
        (cv2.TM_SQDIFF_NORMED, 0.3),    # 30% peso
    ]
    
    total_score = 0.0
    
    for method, weight in methods:
        try:
            result = cv2.matchTemplate(img, template, method)
            
            if method == cv2.TM_SQDIFF_NORMED:
                score = 1.0 - result.min()  # Invertir (menor es mejor)
            else:
                score = result.max()
            
            total_score += score * weight
        except:
            pass
    
    return total_score

def recognize_rank(warp):
    """
    Reconoce el número/letra de una carta
    """
    print("\n=== RECONOCIMIENTO DE RANK ===")
    
    # Definir ROIs (ajustados para capturar mejor el número)
    roi_configs = [
        {'y1': 5, 'y2': 100, 'x1': 5, 'x2': 80, 'name': 'main'},
        {'y1': 10, 'y2': 95, 'x1': 10, 'x2': 75, 'name': 'centered'},
        {'y1': 0, 'y2': 105, 'x1': 0, 'x2': 85, 'name': 'expanded'},
    ]
    
    best_rank = None
    best_score = -1
    
    for i, config in enumerate(roi_configs):
        try:
            roi = warp[config['y1']:config['y2'], config['x1']:config['x2']]
            
            if roi.size == 0:
                continue
            
            # Preprocesar
            processed = preprocess_rank_roi(roi)
            
            # Redimensionar al tamaño de las plantillas (70 alto x 50 ancho)
            resized = cv2.resize(processed, (50, 70), interpolation=cv2.INTER_AREA)
            
            # Guardar debug del primero
            if i == 0:
                cv2.imwrite(f"debug_roi_{config['name']}_original.png", roi)
                cv2.imwrite(f"debug_roi_{config['name']}_processed.png", processed)
                cv2.imwrite(f"debug_roi_{config['name']}_resized.png", resized)
            
            # Matching
            all_scores = {}
            for name, template in RANK_TEMPLATES.items():
                score = match_template_multimethod(resized, template)
                all_scores[name] = score
            
            # Top 5
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            roi_best = sorted_scores[0][0]
            roi_score = sorted_scores[0][1]
            
            if i == 0:
                print(f"\n  [RANK] Top 5 para ROI '{config['name']}':")
                for j, (name, score) in enumerate(sorted_scores[:5]):
                    print(f"    {j+1}. {name}: {score:.4f}")
            
            if roi_score > best_score:
                best_score = roi_score
                best_rank = roi_best
        
        except Exception as e:
            print(f"  [ERROR] ROI '{config['name']}': {e}")
            continue
    
    print(f"\n*** MEJOR MATCH: {best_rank} (score: {best_score:.4f}) ***\n")
    
    return best_rank, best_score

def debug_preprocessing(roi, name="debug"):
    """
    Guarda imágenes en cada paso del preprocesamiento para análisis
    """
    # Paso 1: Original
    cv2.imwrite(f"{name}_1_original.png", roi)
    
    # Paso 2: Grises
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    cv2.imwrite(f"{name}_2_gray.png", gray)
    
    # Paso 3: Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(f"{name}_3_blurred.png", blurred)
    
    # Paso 4: Threshold adaptativo
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite(f"{name}_4_adaptive.png", binary)
    
    # Paso 5: Verificar inversión
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.size
    needs_invert = white_pixels > total_pixels / 2
    
    if needs_invert:
        binary = cv2.bitwise_not(binary)
    cv2.imwrite(f"{name}_5_inverted.png", binary)
    
    # Paso 6: Limpieza
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite(f"{name}_6_opened.png", binary)
    
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite(f"{name}_7_closed.png", binary)
    
    binary = cv2.dilate(binary, kernel, iterations=1)
    cv2.imwrite(f"{name}_8_dilated.png", binary)
    
    # Paso 7: Redimensionado final
    resized = cv2.resize(binary, (50, 70), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{name}_9_final.png", resized)
    
    print(f"\n[DEBUG] Imágenes guardadas con prefijo '{name}'")
    print(f"  - Inversión necesaria: {needs_invert}")
    print(f"  - Píxeles blancos: {white_pixels}/{total_pixels} ({white_pixels/total_pixels*100:.1f}%)")
    
    return resized


    """
    Reconoce el número/letra de una carta desde su imagen warpeada
    """
    print("\n=== RECONOCIMIENTO DE RANK ===")
    
    # Definir múltiples ROIs para probar (por si la carta está ligeramente desalineada)
    roi_configs = [
        {'y1': 10, 'y2': 90, 'x1': 10, 'x2': 70, 'name': 'default'},
        {'y1': 5, 'y2': 95, 'x1': 5, 'x2': 75, 'name': 'expanded'},
        {'y1': 15, 'y2': 85, 'x1': 15, 'x2': 65, 'name': 'contracted'},
        {'y1': 8, 'y2': 92, 'x1': 8, 'x2': 72, 'name': 'shifted'},
    ]
    
    best_rank = None
    best_score = -1
    
    for config in roi_configs:
        try:
            # Extraer ROI
            roi = warp[config['y1']:config['y2'], config['x1']:config['x2']]
            
            if roi.size == 0:
                print(f"  [WARNING] ROI '{config['name']}' está vacío, saltando...")
                continue
            
            # Guardar ROI para debug (opcional)
            # cv2.imwrite(f"debug_rank_roi_{config['name']}.png", roi)
            
            print(f"\n  Probando ROI '{config['name']}'...")
            
            # Hacer el matching
            rank_name, score = match_rank(roi)
            
            print(f"  Resultado: {rank_name} (score: {score:.4f})")
            
            # Actualizar mejor resultado
            if score > best_score:
                best_score = score
                best_rank = rank_name
        
        except Exception as e:
            print(f"  [ERROR] Error con ROI '{config['name']}': {e}")
            continue
    
    print(f"\n*** MEJOR MATCH: {best_rank} (score: {best_score:.4f}) ***\n")
    
    return best_rank, best_score