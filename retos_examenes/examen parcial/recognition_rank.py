import cv2
import numpy as np
import os

def augment_template(binary, name):
    """
    Genera múltiples versiones de un template con diferentes grosores
    """
    versions = {}
    
    # Versión original
    versions[f"{name}_original"] = binary.copy()
    
    # Versión adelgazada (erosión)
    kernel_thin = np.ones((2, 2), np.uint8)
    thinned = cv2.erode(binary, kernel_thin, iterations=1)
    versions[f"{name}_thin"] = thinned
    
    # Versión engrosada (dilatación)
    kernel_thick = np.ones((2, 2), np.uint8)
    thickened = cv2.dilate(binary, kernel_thick, iterations=1)
    versions[f"{name}_thick"] = thickened
    
    # Versión muy engrosada (para templates muy finos)
    kernel_thick2 = np.ones((3, 3), np.uint8)
    very_thick = cv2.dilate(binary, kernel_thick2, iterations=1)
    versions[f"{name}_vthick"] = very_thick
    
    return versions

def load_rank_templates(path='templates/ranks', size=(40, 40)):
    """Carga plantillas con MÚLTIPLES VARIACIONES de grosor"""
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
            continue
        
        rank_name = os.path.splitext(name)[0]
        
        # Detección automática de fondo
        mean_val = cv2.mean(img)[0]
        
        if mean_val > 127:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Limpieza básica
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        if binary.shape != size:
            binary = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
        
        # Generar múltiples versiones
        augmented = augment_template(binary, rank_name)
        
        # Añadir todas las versiones
        for aug_name, aug_template in augmented.items():
            templates[aug_name] = aug_template
            cv2.imwrite(f"debug_template_{aug_name}.png", aug_template)
        
        print(f"[LOAD] '{rank_name}': Generadas {len(augmented)} versiones")
    
    print(f"\n[INIT] Total plantillas (con variaciones): {len(templates)}")
    return templates

RANK_TEMPLATES = load_rank_templates()

def preprocess_rank_roi(roi):
    """Preprocesamiento con MÁS variaciones"""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    versions = []
    
    # 1. Otsu básico
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(('otsu', otsu))
    
    # 2. CLAHE + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    _, clahe_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    versions.append(('clahe_otsu', clahe_otsu))
    
    # 3-5. Threshold fijo con diferentes valores
    for thresh_val, name in [(155, 'fixed_155'), (170, 'fixed_170'), (185, 'fixed_185')]:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, fixed = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        versions.append((name, fixed))
    
    # 6. Adaptativo
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    versions.append(('adaptive', adaptive))
    
    # 7. Con erosión (adelgazar)
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(otsu, kernel, iterations=1)
    versions.append(('otsu_thin', eroded))
    
    # 8. Con dilatación (engrosar)
    dilated = cv2.dilate(otsu, kernel, iterations=1)
    versions.append(('otsu_thick', dilated))
    
    return versions

def extract_main_symbol_simple(binary, target_size=(40, 40)):
    """
    Extracción SIMPLE y robusta
    """
    # Componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    
    if num_labels <= 1:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if len(areas) == 0:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    # Tomar componentes significativos (>15% del máximo)
    max_area = areas.max()
    valid_components = []
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max_area * 0.15:
            valid_components.append(i)
    
    # Máscara con componentes válidos
    mask = np.zeros_like(binary)
    for comp_idx in valid_components:
        mask[labels == comp_idx] = 255
    
    # Bounding box
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    # Padding pequeño
    padding = 3
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(mask.shape[1] - x, w + 2*padding)
    h = min(mask.shape[0] - y, h + 2*padding)
    
    cropped = mask[y:y+h, x:x+w]
    
    if cropped.size == 0:
        return cv2.resize(binary, target_size, interpolation=cv2.INTER_AREA)
    
    # Redimensionar manteniendo aspect ratio
    ch, cw = cropped.shape
    scale = min(target_size[0] / ch, target_size[1] / cw) * 0.88
    
    new_h = max(1, int(ch * scale))
    new_w = max(1, int(cw * scale))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Canvas centrado
    canvas = np.zeros(target_size, dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def match_rank_template(img, template):
    """Matching simple pero efectivo"""
    if img.shape != template.shape:
        img = cv2.resize(img, template.shape[::-1], interpolation=cv2.INTER_AREA)
    
    scores = []
    
    # Template matching
    try:
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        scores.append(max(0, result.max()))
    except:
        scores.append(0.0)
    
    try:
        result = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        scores.append(max(0, result.max()))
    except:
        scores.append(0.0)
    
    # IoU
    try:
        intersection = cv2.bitwise_and(img, template)
        union = cv2.bitwise_or(img, template)
        
        inter_count = cv2.countNonZero(intersection)
        union_count = cv2.countNonZero(union)
        
        iou = inter_count / union_count if union_count > 0 else 0
        scores.append(iou)
    except:
        scores.append(0.0)
    
    # Promedio
    return sum(scores) / len(scores) if scores else 0.0

def recognize_rank(warp):
    """Reconoce el rank con sistema de votación robusto"""
    print("\n" + "="*60)
    print("RECONOCIMIENTO DE RANK")
    print("="*60)
    
    cv2.imwrite("debug_warp_full.png", warp)
    
    roi = warp[8:92, 8:68]
    cv2.imwrite("debug_roi_original.png", roi)
    
    preprocessed_versions = preprocess_rank_roi(roi)
    
    for idx, (name, img) in enumerate(preprocessed_versions, 1):
        cv2.imwrite(f"debug_{idx}_{name}.png", img)
    
    # Sistema de votación
    vote_scores = {}  # rank_base -> lista de scores
    
    for v_idx, (v_name, binary) in enumerate(preprocessed_versions):
        cleaned = extract_main_symbol_simple(binary, target_size=(40, 40))
        
        if v_idx == 0:
            cv2.imwrite("debug_cleaned_main.png", cleaned)
        
        # Matching contra todas las plantillas (con variaciones)
        for template_name, template in RANK_TEMPLATES.items():
            score = match_rank_template(cleaned, template)
            
            # Extraer rank base (sin sufijos _original, _thin, etc.)
            rank_base = template_name.split('_')[0]
            
            if rank_base not in vote_scores:
                vote_scores[rank_base] = []
            
            vote_scores[rank_base].append(score)
    
    # Calcular score final por rank (promedio de los 5 mejores)
    final_scores = {}
    for rank_base, scores in vote_scores.items():
        top_scores = sorted(scores, reverse=True)[:5]
        final_scores[rank_base] = sum(top_scores) / len(top_scores)
    
    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*60}")
    print("TOP 10 RESULTADOS:")
    print(f"{'='*60}")
    for i, (name, score) in enumerate(sorted_results[:10], 1):
        marker = " ← ELEGIDO" if i == 1 else ""
        print(f"  {i:2d}. {name:8s} : {score:.4f}{marker}")
    
    if sorted_results:
        best_rank = sorted_results[0][0]
        best_score = sorted_results[0][1]
        
        print(f"\n{'='*60}")
        print(f"*** RESULTADO FINAL: {best_rank} (confianza: {best_score:.4f}) ***")
        print(f"{'='*60}\n")
        
        return best_rank, best_score
    
    return None, 0.0