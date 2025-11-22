import cv2
import numpy as np
import os

def load_rank_templates(path='templates/ranks', target_size=(50, 70)):
    """
    Carga templates y los normaliza INMEDIATAMENTE al mismo tamaño
    """
    templates = {}
    
    if not os.path.exists(path):
        print(f"[ERROR] No existe el directorio: {path}")
        return templates
    
    for filename in os.listdir(path):
        if not filename.lower().endswith('.png'):
            continue
        
        filepath = os.path.join(path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        rank_name = os.path.splitext(filename)[0]
        
        # Detectar fondo y binarizar
        mean_val = cv2.mean(img)[0]
        
        if mean_val > 127:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Limpiar ruido
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # NORMALIZAR INMEDIATAMENTE
        normalized = normalize_symbol(binary, target_size)
        
        templates[rank_name] = normalized
        
        # Guardar para debug
        cv2.imwrite(f"template_{rank_name}_normalized.png", normalized)
        
        print(f"[LOAD] '{rank_name}': normalizado a {target_size}")
    
    print(f"[INIT] Total templates: {len(templates)}\n")
    return templates

def normalize_symbol(binary, target_size=(50, 70)):
    """
    Normaliza un símbolo: extrae, centra y redimensiona manteniendo aspect ratio
    """
    # Encontrar componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    if num_labels <= 1:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if len(areas) == 0:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    # Tomar componentes significativos (>10% del máximo)
    max_area = areas.max()
    valid_components = []
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max_area * 0.10:
            valid_components.append(i)
    
    # Crear máscara con componentes válidos
    mask = np.zeros_like(binary)
    for comp_idx in valid_components:
        mask[labels == comp_idx] = 255
    
    # Bounding box
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Padding mínimo
    pad = 3
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(mask.shape[1] - x, w + 2*pad)
    h = min(mask.shape[0] - y, h + 2*pad)
    
    cropped = mask[y:y+h, x:x+w]
    
    if cropped.size == 0:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    # Redimensionar manteniendo aspect ratio
    ch, cw = cropped.shape
    scale = min(target_size[1] / ch, target_size[0] / cw) * 0.88
    
    new_h = max(1, int(ch * scale))
    new_w = max(1, int(cw * scale))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas centrado
    canvas = np.zeros(target_size[::-1], dtype=np.uint8)
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

RANK_TEMPLATES = load_rank_templates()

def preprocess_rank_roi(roi):
    """Preprocesamiento SIMPLIFICADO"""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    versions = []
    
    # 1. CLAHE + Otsu (el mejor en general)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(('clahe_otsu', clahe_otsu))
    
    # 2. Otsu directo
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(('otsu', otsu))
    
    # 3. Adaptativo
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    versions.append(('adaptive', adaptive))
    
    return versions

def detect_10(binary):
    """
    Detecta si hay DOS componentes separados (característico del "10")
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    if num_labels <= 1:
        return False
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if len(areas) < 2:
        return False
    
    # Ordenar por área
    sorted_areas = sorted(areas, reverse=True)
    
    # Si los dos componentes más grandes son similares (>30% del mayor)
    if len(sorted_areas) >= 2:
        if sorted_areas[1] >= sorted_areas[0] * 0.30:
            # Y además hay 2 componentes significativos
            significant_comps = sum(1 for area in areas if area >= sorted_areas[0] * 0.30)
            
            if significant_comps >= 2:
                # Verificar que están separados horizontalmente
                valid_components = []
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] >= sorted_areas[0] * 0.30:
                        valid_components.append(i)
                
                if len(valid_components) >= 2:
                    # Obtener bounding boxes
                    x_positions = []
                    for comp_idx in valid_components[:2]:
                        x = stats[comp_idx, cv2.CC_STAT_LEFT]
                        x_positions.append(x)
                    
                    # Si están separados horizontalmente (diferencia > 5px)
                    if abs(x_positions[0] - x_positions[1]) > 5:
                        return True
    
    return False

def compare_direct(img, template):
    """
    Comparación DIRECTA: asume mismo tamaño
    """
    if img.shape != template.shape:
        img = cv2.resize(img, (template.shape[1], template.shape[0]), 
                        interpolation=cv2.INTER_AREA)
    
    # 1. Template Matching CCOEFF_NORMED (50%)
    try:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        score_ccoeff = max(0.0, result.max())
    except:
        score_ccoeff = 0.0
    
    # 2. Template Matching CCORR_NORMED (30%)
    try:
        result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        score_ccorr = max(0.0, result.max())
    except:
        score_ccorr = 0.0
    
    # 3. IoU (Intersection over Union) (20%)
    try:
        intersection = cv2.bitwise_and(img, template)
        union = cv2.bitwise_or(img, template)
        
        inter_count = cv2.countNonZero(intersection)
        union_count = cv2.countNonZero(union)
        
        iou = inter_count / union_count if union_count > 0 else 0.0
    except:
        iou = 0.0
    
    # Score final ponderado
    final_score = 0.50 * score_ccoeff + 0.30 * score_ccorr + 0.20 * iou
    
    return final_score

def recognize_rank(warp):
    """
    Reconocimiento SIMPLE Y DIRECTO con detección especial del 10
    """
    print("\n" + "="*70)
    print("RECONOCIMIENTO DE RANK (CON DETECCIÓN DE 10)")
    print("="*70)
    
    # ROI del rank
    roi = warp[10:100, 10:80]
    
    print(f"ROI size: {roi.shape}")
    cv2.imwrite("debug_rank_roi.png", roi)
    
    # Preprocesar
    preprocessed = preprocess_rank_roi(roi)
    
    for idx, (name, img) in enumerate(preprocessed, 1):
        cv2.imwrite(f"debug_rank_{idx}_{name}.png", img)
    
    # DETECCIÓN ESPECIAL DEL 10
    is_ten_votes = 0
    for version_name, binary in preprocessed:
        if detect_10(binary):
            is_ten_votes += 1
    
    if is_ten_votes >= 2:
        print("\n" + "!"*70)
        print("*** DETECTADO: DOS COMPONENTES → Es un 10 ***")
        print("!"*70 + "\n")
        return "10", 1.0
    
    # Votación simple para el resto
    vote_scores = {}
    
    for version_name, binary in preprocessed:
        # Normalizar de la misma forma que los templates
        normalized = normalize_symbol(binary, target_size=(50, 70))
        
        if version_name == 'clahe_otsu':
            cv2.imwrite("debug_rank_normalized.png", normalized)
        
        # Comparar DIRECTAMENTE contra cada template
        for rank_name, template in RANK_TEMPLATES.items():
            # Excluir el 10 del matching normal
            if rank_name == '10':
                continue
            
            score = compare_direct(normalized, template)
            
            if rank_name not in vote_scores:
                vote_scores[rank_name] = []
            
            vote_scores[rank_name].append(score)
    
    # Score final: promedio simple
    final_scores = {}
    for rank_name, scores in vote_scores.items():
        final_scores[rank_name] = sum(scores) / len(scores) if scores else 0.0
    
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 10 RESULTADOS:")
    print(f"{'='*70}")
    for i, (name, score) in enumerate(sorted_results[:10], 1):
        marker = " ← ELEGIDO" if i == 1 else ""
        bar = "█" * int(score * 50)
        print(f"  {i:2d}. {name:8s} : {score:.4f} {bar}{marker}")
    
    if sorted_results:
        best_rank = sorted_results[0][0]
        best_score = sorted_results[0][1]
        
        print(f"\n{'='*70}")
        print(f"*** RANK: {best_rank} (confianza: {best_score:.4f}) ***")
        print(f"{'='*70}\n")
        
        return best_rank, best_score
    
    return None, 0.0