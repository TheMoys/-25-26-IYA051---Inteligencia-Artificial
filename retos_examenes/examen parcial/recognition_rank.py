import cv2
import numpy as np
import os

def load_rank_templates(path='templates/ranks', size=(50, 70)):
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
        
        # Binarizar: símbolo BLANCO sobre fondo NEGRO
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Verificar si el símbolo es negro (invertir si es necesario)
        white_pixels = cv2.countNonZero(binary)
        if white_pixels < binary.size / 2:
            binary = cv2.bitwise_not(binary)
        
        # Redimensionar
        resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
        
        # Limpiar ruido
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(resized, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        rank_name = os.path.splitext(name)[0]
        templates[rank_name] = clean
        
        print(f"[LOAD] Template '{rank_name}' cargado: {clean.shape}")
    
    print(f"\n[INIT] Total plantillas cargadas: {len(templates)}")
    return templates

# Cargar plantillas globales
RANK_TEMPLATES = load_rank_templates()

def preprocess_rank_roi(roi):
    """
    Preprocesa el ROI del rank: convierte a binario con símbolo BLANCO sobre fondo NEGRO
    """
    # Convertir a grises si es necesario
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # Aplicar blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold adaptativo (mejor para diferentes iluminaciones)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # Determinar si necesitamos invertir
    white_pixels = cv2.countNonZero(binary)
    total_pixels = binary.size
    
    # Si más del 50% es blanco, el fondo es blanco -> invertir
    if white_pixels > total_pixels / 2:
        binary = cv2.bitwise_not(binary)
    
    # Limpieza morfológica
    kernel = np.ones((3, 3), np.uint8)
    
    # Eliminar ruido pequeño
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Cerrar huecos
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilatar ligeramente para hacer el símbolo más sólido
    binary = cv2.dilate(binary, kernel, iterations=1)
    
    return binary

def match_template_multimethod(img, template):
    """
    Aplica múltiples métodos de template matching y retorna el mejor score
    """
    methods = [
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF_NORMED
    ]
    
    scores = []
    
    for method in methods:
        try:
            result = cv2.matchTemplate(img, template, method)
            
            if method == cv2.TM_SQDIFF_NORMED:
                # Para este método, menor es mejor, así que invertir
                score = 1.0 - result.min()
            else:
                score = result.max()
            
            scores.append(score)
        except:
            scores.append(0.0)
    
    # Retornar el promedio de los scores
    return np.mean(scores)

def match_rank(roi):
    """
    Encuentra el mejor match del rank usando template matching
    """
    if len(RANK_TEMPLATES) == 0:
        print("[ERROR] No hay plantillas cargadas!")
        return None, 0.0
    
    # Preprocesar ROI
    processed = preprocess_rank_roi(roi)
    
    # Redimensionar al tamaño de las plantillas
    resized = cv2.resize(processed, (50, 70), interpolation=cv2.INTER_AREA)
    
    best_name = None
    best_score = -1
    all_scores = {}
    
    # Probar con cada plantilla
    for name, template in RANK_TEMPLATES.items():
        # Score usando múltiples métodos
        score = match_template_multimethod(resized, template)
        all_scores[name] = score
        
        if score > best_score:
            best_score = score
            best_name = name
    
    # Mostrar top 5 candidatos
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n  [RANK] Top 5 candidatos:")
    for i, (name, score) in enumerate(sorted_scores[:5]):
        marker = " <-- ELEGIDO" if name == best_name else ""
        print(f"    {i+1}. {name}: {score:.4f}{marker}")
    
    return best_name, best_score

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

def recognize_rank(warp):
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