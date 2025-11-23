import cv2
import numpy as np
import os

def load_rank_templates(path='templates/ranks', target_size=(50, 70)):
    """
    Carga templates con PRIORIDADES (primary → secondary)
    Cada rank puede tener múltiples variantes
    """
    templates = {
        'primary': {},
        'secondary': {}
    }
    
    for priority in ['primary', 'secondary']:
        priority_path = os.path.join(path, priority)
        
        if not os.path.exists(priority_path):
            print(f"[WARNING] No existe: {priority_path}")
            continue
        
        print(f"\n[LOADING {priority.upper()}]")
        
        for filename in os.listdir(priority_path):
            if not filename.lower().endswith('.png'):
                continue
            
            filepath = os.path.join(priority_path, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Extraer nombre del rank (A, 2, K, etc.)
            # Formato: "A.png" o "A_v2.png" → rank_name = "A"
            rank_name = filename.split('_')[0].split('.')[0]
            
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
            
            # Guardar múltiples variantes por rank
            if rank_name not in templates[priority]:
                templates[priority][rank_name] = []
            
            templates[priority][rank_name].append(normalized)
            
            # Guardar para debug
            variant_num = len(templates[priority][rank_name])
            cv2.imwrite(f"template_{priority}_{rank_name}_v{variant_num}_normalized.png", normalized)
            
            print(f"  [LOAD] {priority}/{rank_name} (variante {variant_num}): {filename}")
    
    # Resumen
    total_primary = sum(len(v) for v in templates['primary'].values())
    total_secondary = sum(len(v) for v in templates['secondary'].values())
    
    print(f"\n[SUMMARY]")
    print(f"  Primary templates: {total_primary}")
    print(f"  Secondary templates: {total_secondary}")
    print(f"  Total: {total_primary + total_secondary}\n")
    
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
    
        # Votación CON MÚLTIPLES VARIANTES
    vote_scores = {}
    
    for version_name, binary in preprocessed:
        # Normalizar de la misma forma que los templates
        normalized = normalize_symbol(binary, target_size=(50, 70))
        
        if version_name == 'clahe_otsu':
            cv2.imwrite("debug_rank_normalized.png", normalized)
        
        # PASO 1: Comparar contra PRIMARY templates
        for rank_name, template_list in RANK_TEMPLATES['primary'].items():
            # Excluir el 10 del matching normal
            if rank_name == '10':
                continue
            
            # Comparar contra TODAS las variantes de este rank
            max_score = 0.0
            for template in template_list:
                score = compare_direct(normalized, template)
                max_score = max(max_score, score)
            
            if rank_name not in vote_scores:
                vote_scores[rank_name] = {'primary': [], 'secondary': []}
            
            vote_scores[rank_name]['primary'].append(max_score)
        
        # PASO 2: Comparar contra SECONDARY templates (si existen)
        if len(RANK_TEMPLATES['secondary']) > 0:
            for rank_name, template_list in RANK_TEMPLATES['secondary'].items():
                if rank_name == '10':
                    continue
                
                max_score = 0.0
                for template in template_list:
                    score = compare_direct(normalized, template)
                    max_score = max(max_score, score)
                
                if rank_name not in vote_scores:
                    vote_scores[rank_name] = {'primary': [], 'secondary': []}
                
                vote_scores[rank_name]['secondary'].append(max_score)
    
    # Score final: PROMEDIO PONDERADO (primary 60%, secondary 40%)
    final_scores = {}
    
    for rank_name, scores_dict in vote_scores.items():
        primary_scores = scores_dict['primary']
        secondary_scores = scores_dict['secondary']
        
        # Calcular promedios
        primary_avg = sum(primary_scores) / len(primary_scores) if primary_scores else 0.0
        secondary_avg = sum(secondary_scores) / len(secondary_scores) if secondary_scores else 0.0
        
        # Ponderación: Primary 60%, Secondary 40%
        if secondary_avg > 0:
            final_score = 0.60 * primary_avg + 0.40 * secondary_avg
        else:
            # Si no hay secondary templates, usar solo primary
            final_score = primary_avg
        
        final_scores[rank_name] = {
            'final': final_score,
            'primary': primary_avg,
            'secondary': secondary_avg
        }
    
    # Ordenar por score final
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1]['final'], reverse=True)
    
    print(f"\n{'='*70}")
    print("TOP 10 RESULTADOS (con desglose primary/secondary):")
    print(f"{'='*70}")
    
    for i, (name, scores) in enumerate(sorted_results[:10], 1):
        marker = " ← ELEGIDO" if i == 1 else ""
        bar = "█" * int(scores['final'] * 50)
        
        # Mostrar desglose
        primary_str = f"P:{scores['primary']:.3f}"
        secondary_str = f"S:{scores['secondary']:.3f}" if scores['secondary'] > 0 else "S:---"
        
        print(f"  {i:2d}. {name:8s} : {scores['final']:.4f} [{primary_str}, {secondary_str}] {bar}{marker}")
    
    if sorted_results:
        best_rank = sorted_results[0][0]
        best_scores = sorted_results[0][1]
        
        print(f"\n{'='*70}")
        print(f"*** RANK: {best_rank} ***")
        print(f"  Confianza final: {best_scores['final']:.4f}")
        print(f"  Primary: {best_scores['primary']:.4f}")
        print(f"  Secondary: {best_scores['secondary']:.4f}")
        print(f"{'='*70}\n")
        
        return best_rank, best_scores['final']
    
    return None, 0.0