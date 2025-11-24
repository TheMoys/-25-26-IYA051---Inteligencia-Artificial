import cv2
import numpy as np
import os

def load_suit_templates(path='templates/suits', size=(70,70)):
    """
    Carga templates de palos con PRIORIDADES (primary → secondary)
    Cada palo puede tener múltiples variantes
    """
    templates = {
        'red': {
            'primary': {},
            'secondary': {}
        },
        'black': {
            'primary': {},
            'secondary': {}
        }
    }
    
    for color in ['red', 'black']:
        color_path = os.path.join(path, color)
        
        if not os.path.exists(color_path):
            print(f"[WARNING] No existe: {color_path}")
            continue
        
        for priority in ['primary', 'secondary']:
            priority_path = os.path.join(color_path, priority)
            
            if not os.path.exists(priority_path):
                print(f"[WARNING] No existe: {priority_path}")
                continue
            
            print(f"\n[LOADING {color.upper()} {priority.upper()}]")
            
            for filename in os.listdir(priority_path):
                if not filename.lower().endswith('.png'):
                    continue
                
                filepath = os.path.join(priority_path, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    continue
                
                # Extraer nombre del palo (diamonds, hearts, spades, clubs)
                # Formato: "diamonds.png" o "diamonds_v2.png" → suit_name = "diamonds"
                base_name = os.path.splitext(filename)[0].lower()
                suit_name = base_name.split('_')[0]  # Obtener solo "diamonds", "hearts", etc.
                
                # Normalizar nombres alternativos
                if suit_name in ['corazones', 'diamantes', 'oros']:
                    suit_name = 'hearts' if suit_name == 'corazones' else 'diamonds'
                elif suit_name in ['picas', 'treboles', 'espadas', 'bastos']:
                    suit_name = 'spades' if suit_name in ['picas', 'espadas'] else 'clubs'
                
                # Redimensionar
                img = cv2.resize(img, size)
                
                # Binarizar
                _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # Limpiar y normalizar
                kernel = np.ones((3,3), np.uint8)
                img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
                img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                # Guardar múltiples variantes por palo
                if suit_name not in templates[color][priority]:
                    templates[color][priority][suit_name] = []
                
                templates[color][priority][suit_name].append(img_bin)
                
                # Guardar para debug
                variant_num = len(templates[color][priority][suit_name])
                cv2.imwrite(f"template_suit_{color}_{priority}_{suit_name}_v{variant_num}.png", img_bin)
                
                print(f"  [LOAD] {color}/{priority}/{suit_name} (variante {variant_num}): {filename}")
    
    # Resumen
    total_red_pri = sum(len(v) for v in templates['red']['primary'].values())
    total_red_sec = sum(len(v) for v in templates['red']['secondary'].values())
    total_black_pri = sum(len(v) for v in templates['black']['primary'].values())
    total_black_sec = sum(len(v) for v in templates['black']['secondary'].values())
    
    print(f"\n[SUMMARY SUITS]")
    print(f"  Red primary: {total_red_pri}")
    print(f"  Red secondary: {total_red_sec}")
    print(f"  Black primary: {total_black_pri}")
    print(f"  Black secondary: {total_black_sec}")
    print(f"  Total: {total_red_pri + total_red_sec + total_black_pri + total_black_sec}\n")
    
    return templates

# Cargar plantillas al inicio
SUIT_TEMPLATES = load_suit_templates()

def is_red_suit(suit_roi):
    """
    Determina si el palo es rojo (corazones/diamantes) o negro (picas/tréboles)
    """
    gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    _, symbol_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3,3), np.uint8)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_OPEN, kernel)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)
    
    if cv2.countNonZero(symbol_mask) < 50:
        return False
    
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(suit_roi)
    
    r_symbol = cv2.bitwise_and(r, r, mask=symbol_mask)
    b_symbol = cv2.bitwise_and(b, b, mask=symbol_mask)
    g_symbol = cv2.bitwise_and(g, g, mask=symbol_mask)
    
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_and(mask_red, symbol_mask)
    
    red_pixels_hsv = cv2.countNonZero(mask_red)
    symbol_pixels = cv2.countNonZero(symbol_mask)
    
    ratio_hsv = red_pixels_hsv / symbol_pixels if symbol_pixels > 0 else 0
    
    mask_rgb = np.zeros(r.shape, dtype=np.uint8)
    mask_rgb[(r > b + 20) & (r > g) & (r > 80)] = 255
    mask_rgb = cv2.bitwise_and(mask_rgb, symbol_mask)
    red_pixels_rgb = cv2.countNonZero(mask_rgb)
    ratio_rgb = red_pixels_rgb / symbol_pixels if symbol_pixels > 0 else 0
    
    mean_r = np.sum(r_symbol) / symbol_pixels if symbol_pixels > 0 else 0
    mean_b = np.sum(b_symbol) / symbol_pixels if symbol_pixels > 0 else 0
    mean_g = np.sum(g_symbol) / symbol_pixels if symbol_pixels > 0 else 0
    
    is_red_hsv = ratio_hsv > 0.08
    is_red_rgb = ratio_rgb > 0.12
    is_red_channel = (mean_r - mean_b) > 15 and (mean_r - mean_g) > 10
    
    votes = sum([is_red_hsv, is_red_rgb, is_red_channel])
    
    return votes >= 2

def analyze_convexity_defects(img):
    """
    Analiza los defectos de convexidad para distinguir formas
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0, 0, []
    
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)
    
    if len(hull) > 3 and len(cnt) > 3:
        defects = cv2.convexityDefects(cnt, hull)
        
        if defects is not None:
            # Contar defectos significativos
            significant_defects = 0
            max_depth = 0
            defect_depths = []
            
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0  # Convertir a píxeles
                
                defect_depths.append(depth)
                if depth > max_depth:
                    max_depth = depth
                
                # Defecto significativo si la profundidad es > 5 píxeles
                if depth > 5:
                    significant_defects += 1
            
            return significant_defects, max_depth, defect_depths
    
    return 0, 0, []

def detect_top_notch(img):
    """
    Detecta hendidura superior - MUY PERMISIVA para corazones pequeños
    """
    h, w = img.shape
    
    # Analizar el 30% superior
    top_section = img[:int(h*0.30), :]
    
    # Buscar la hendidura
    top_inverted = cv2.bitwise_not(top_section)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(top_inverted, connectivity=8)
    
    notch_found = False
    notch_depth = 0
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        center_x = x + width / 2
        
        # CRITERIOS MUY RELAJADOS
        is_centered = (w * 0.25) < center_x < (w * 0.75)  # Rango amplio
        is_at_top = y < int(h * 0.18)  # Más abajo
        
        # Tamaños adaptativos
        min_area = max(15, w * h * 0.008)  # Al menos 0.8% del área O 15 píxeles
        min_depth = max(4, h * 0.06)  # Al menos 6% de altura O 4 píxeles
        
        significant_size = area > min_area
        significant_depth = height > min_depth
        
        # Muy permisivo con la forma
        is_notch_shape = height > width * 0.4  # Solo que sea algo vertical
        
        if is_centered and is_at_top and significant_size and significant_depth and is_notch_shape:
            notch_found = True
            notch_depth = max(notch_depth, height)
    
    return notch_found, notch_depth

def detect_rounded_top(img):
    """
    Detecta si la parte superior tiene formas redondeadas (los dos lóbulos del corazón)
    """
    h, w = img.shape
    
    # Analizar mitad superior
    top_half = img[:h//2, :]
    
    # Detectar círculos/elipses en la parte superior
    circles = cv2.HoughCircles(
        top_half,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=int(w*0.3),
        param1=50,
        param2=15,
        minRadius=int(w*0.2),
        maxRadius=int(w*0.45)
    )
    
    has_rounded_lobes = circles is not None and len(circles[0]) >= 2
    
    return has_rounded_lobes

def detect_bottom_point(img):
    """
    Detecta una punta aguda en la parte inferior central
    """
    h, w = img.shape
    
    # Analizar tercio inferior
    bottom_third = img[2*h//3:, :]
    
    # Buscar el punto más bajo del contorno
    contours, _ = cv2.findContours(bottom_third, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, 0
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Encontrar el punto más bajo (mayor Y)
    lowest_point = tuple(cnt[cnt[:, :, 1].argmax()][0])
    
    # Verificar si está centrado horizontalmente
    center_x = w // 2
    point_x = lowest_point[0]
    is_centered = abs(point_x - center_x) < (w * 0.3)
    
    # Verificar que hay masa concentrada en ese punto
    point_area = cv2.countNonZero(bottom_third[:, max(0, point_x-5):min(w, point_x+5)])
    
    has_point = is_centered and point_area > 30
    
    return has_point, point_area

def detect_diamond_pattern(img):
    """
    Detecta diamante: FORMA SIMPLE, 4 ESQUINAS, MUY CONVEXO
    """
    h, w = img.shape
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 1. CRITERIO FUNDAMENTAL: Aproximación con 4 vértices
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)
    
    # Probar con diferentes epsilons si falla
    if num_vertices != 4:
        for eps_mult in [0.015, 0.025, 0.03]:
            epsilon = eps_mult * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:
                num_vertices = 4
                break
    
    has_4_vertices = (num_vertices == 4)
    
    # 2. Solidez ALTA (muy convexo, sin hendiduras)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(cnt)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # 3. Aspect ratio cercano a 1 (diamante es aproximadamente cuadrado)
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
    
    # 4. Extensión (qué tanto llena su bounding box)
    extent = float(area) / (w_box * h_box) if (w_box * h_box) > 0 else 0
    
    # Debug detallado
    print(f"    [DIAMOND DEBUG] Vértices: {num_vertices} (req: 4)")
    print(f"    [DIAMOND DEBUG] Solidez: {solidity:.3f} (req: >0.88)")
    print(f"    [DIAMOND DEBUG] Aspect ratio: {aspect_ratio:.3f} (req: 0.6-1.4)")
    print(f"    [DIAMOND DEBUG] Extent: {extent:.3f} (req: >0.45)")
    
    # SCORE: Todos los criterios deben cumplirse
    score = 0.0
    
    if has_4_vertices:
        score += 0.50  # Base fundamental
        
        # Solidez muy alta
        if solidity > 0.92:
            score += 0.25
        elif solidity > 0.88:
            score += 0.15
        
        # Aspect ratio razonable (no muy alargado)
        if 0.6 < aspect_ratio < 1.4:
            score += 0.15
        
        # Llena bien su bounding box
        if extent > 0.50:
            score += 0.10
        elif extent > 0.45:
            score += 0.05
        
        print(f"    [DIAMOND DEBUG] ✓ Score: {score:.3f}")
    else:
        print(f"    [DIAMOND DEBUG] ✗ RECHAZADO: No tiene 4 vértices")
    
    is_diamond = has_4_vertices and solidity > 0.88 and 0.6 < aspect_ratio < 1.4
    
    return is_diamond, score

def detect_heart_pattern(img):
    """
    Detecta corazón: FORMA COMPLEJA, SIN 4 VÉRTICES
    VERSIÓN ULTRA PERMISIVA
    """
    h, w = img.shape
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 1. Contar vértices con VARIOS epsilons
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    num_vertices = len(approx)
    
    # También probar con epsilon más alto (más suavizado)
    epsilon_high = 0.04 * cv2.arcLength(cnt, True)
    approx_smooth = cv2.approxPolyDP(cnt, epsilon_high, True)
    num_vertices_smooth = len(approx_smooth)
    
    # Corazón tiene forma compleja (NO 4 vértices limpios)
    has_many_vertices = num_vertices > 6  # Bajado de >4 a >6 para ser más específico
    
    # 2. Solidez
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(cnt)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # 3. Hendidura
    has_notch, notch_depth = detect_top_notch(img)
    
    # 4. Defectos
    defects_count, max_depth, _ = analyze_convexity_defects(img)
    
    # 5. Aspect ratio
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
    
    # 6. Extensión (corazones llenan más su bounding box que diamantes)
    extent = float(area) / (w_box * h_box) if (w_box * h_box) > 0 else 0
    
    # Debug detallado
    print(f"    [HEART DEBUG] Vértices: {num_vertices} (smooth: {num_vertices_smooth})")
    print(f"    [HEART DEBUG] Solidez: {solidity:.3f}, Extent: {extent:.3f}")
    print(f"    [HEART DEBUG] Hendidura: {has_notch} depth={notch_depth}")
    print(f"    [HEART DEBUG] Defectos: {defects_count}, Aspect: {aspect_ratio:.3f}")
    
    # SCORE con múltiples caminos
    score = 0.0
    
    # BLOQUEO: Si tiene exactamente 4 vértices suaves, probablemente es diamante
    if num_vertices_smooth == 4 and solidity > 0.90:
        print(f"    [HEART DEBUG] ✗ BLOQUEADO: Diamante (4v suaves + solidez muy alta)")
        return False, 0.0
    
    # CAMINO 1: Hendidura detectada (característico de corazón)
    if has_notch and notch_depth > 5:
        score += 0.60
        if has_many_vertices:
            score += 0.20
        if extent > 0.60:
            score += 0.10
        if aspect_ratio > 0.85:
            score += 0.10
        print(f"    [HEART DEBUG] ✓ Hendidura detectada, score: {score:.3f}")
    
    # CAMINO 2: Muchos vértices + NO es diamante limpio
    elif has_many_vertices and num_vertices_smooth > 4:
        score += 0.45
        
        # Bonus por características adicionales
        if extent > 0.60:  # Llena bien su caja
            score += 0.20
        if aspect_ratio > 1.0:  # Más ancho que alto
            score += 0.15
        if defects_count >= 1:
            score += 0.10
        if solidity < 0.95:  # No es PERFECTAMENTE sólido
            score += 0.10
        
        print(f"    [HEART DEBUG] ✓ Forma compleja ({num_vertices}v), score: {score:.3f}")
    
    # CAMINO 3: Por exclusión - claramente NO es diamante
    elif num_vertices > 8 and extent > 0.60:
        score += 0.40
        
        if aspect_ratio > 0.90:
            score += 0.20
        if defects_count >= 1:
            score += 0.15
        
        print(f"    [HEART DEBUG] ~ No es diamante, score: {score:.3f}")
    
    # CAMINO 4: NUEVO - Alto extent + muchos vértices (incluso con alta solidez)
    elif extent > 0.65 and num_vertices > 8:
        score += 0.50
        
        if aspect_ratio > 1.0:
            score += 0.20
        if num_vertices > 10:
            score += 0.15
        
        print(f"    [HEART DEBUG] ✓ Alto extent ({extent:.3f}), score: {score:.3f}")
    
    else:
        print(f"    [HEART DEBUG] ✗ No cumple criterios de corazón")
    
    # Condición flexible
    is_heart = (has_notch and notch_depth > 5) or \
               (has_many_vertices and num_vertices_smooth > 4) or \
               (num_vertices > 8 and extent > 0.60)
    
    # Umbral mínimo de score
    if score < 0.30:
        is_heart = False
    
    return is_heart, score

def detect_spade_pattern(img):
    """
    Detecta pica: FORMA ESPECÍFICA con solidez ALTA
    VERSION REFINADA: Más estricto con solidez
    """
    h, w = img.shape
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 1. Solidez (pica: 0.85-0.92) - MÁS ESTRICTO
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(cnt)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    
    # 2. Defectos de convexidad (pica: 1-4, club: 8+)
    defects_count, max_depth, _ = analyze_convexity_defects(img)
    few_defects = 1 <= defects_count <= 5
    
    # 3. Análisis por FRANJAS HORIZONTALES
    strip_height = h // 5
    strip_widths = []
    
    for i in range(5):
        y_start = i * strip_height
        y_end = min((i + 1) * strip_height, h)
        strip = img[y_start:y_end, :]
        
        strip_projection = np.sum(strip, axis=0)
        if len(strip_projection) > 0 and np.max(strip_projection) > 0:
            strip_projection = strip_projection / np.max(strip_projection)
            high_intensity_cols = np.sum(strip_projection > 0.5)
            total_cols = len(strip_projection)
            width_ratio = high_intensity_cols / total_cols if total_cols > 0 else 0
            strip_widths.append(width_ratio)
        else:
            strip_widths.append(0)
    
    top_width = strip_widths[0] if len(strip_widths) > 0 else 0
    upper_mid_width = strip_widths[1] if len(strip_widths) > 1 else 0
    middle_width = strip_widths[2] if len(strip_widths) > 2 else 0
    lower_mid_width = strip_widths[3] if len(strip_widths) > 3 else 0
    bottom_width = strip_widths[4] if len(strip_widths) > 4 else 0
    
    print(f"    [SPADE STRIPS] Anchos por franja:")
    print(f"      Top (0-20%):      {top_width:.3f}")
    print(f"      Upper-mid (20-40%): {upper_mid_width:.3f}")
    print(f"      Middle (40-60%):    {middle_width:.3f}")
    print(f"      Lower-mid (60-80%): {lower_mid_width:.3f}")
    print(f"      Bottom (80-100%):   {bottom_width:.3f}")
    
    # Criterios de pica
    top_narrower = top_width < middle_width * 0.85
    max_width_in_middle = max(upper_mid_width, middle_width) > max(top_width, lower_mid_width, bottom_width)
    bottom_narrowest = bottom_width < min(top_width, middle_width) * 0.85 if middle_width > 0 else False
    grows_to_middle = upper_mid_width > top_width * 0.90
    shrinks_to_bottom = lower_mid_width < middle_width * 0.90
    
    print(f"    [SPADE PATTERN]")
    print(f"      Top narrower than middle: {top_narrower}")
    print(f"      Max width in middle: {max_width_in_middle}")
    print(f"      Bottom narrowest: {bottom_narrowest}")
    print(f"      Grows to middle: {grows_to_middle}")
    print(f"      Shrinks to bottom: {shrinks_to_bottom}")
    
    # 4. Simetría
    left_half = img[:, :w//2]
    right_half = cv2.flip(img[:, w//2:], 1)
    
    min_width = min(left_half.shape[1], right_half.shape[1])
    left_half = left_half[:, :min_width]
    right_half = right_half[:, :min_width]
    
    difference = cv2.absdiff(left_half, right_half)
    symmetry_score = 1.0 - (np.mean(difference) / 255.0)
    
    is_symmetric = symmetry_score > 0.70
    
    # 5. Aspect ratio
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
    good_aspect = 0.60 < aspect_ratio < 1.30
    
    print(f"    [SPADE DEBUG] Solidez: {solidity:.3f} (req: 0.82-0.93)")
    print(f"    [SPADE DEBUG] Defectos: {defects_count} (req: 1-5)")
    print(f"    [SPADE DEBUG] Simetría: {symmetry_score:.3f}")
    print(f"    [SPADE DEBUG] Aspect: {aspect_ratio:.3f}")
    
    # BLOQUEO CRÍTICO: Solidez baja = NO es pica
    if solidity < 0.78:
        print(f"    [SPADE DEBUG] ✗ BLOQUEADO: Solidez muy baja ({solidity:.3f} < 0.78)")
        print(f"    [SPADE DEBUG]    Probablemente es TRÉBOL (múltiples lóbulos)")
        return False, 0.0
    
    # Scoring
    score = 0.0
    
    if top_narrower and max_width_in_middle:
        score += 0.20
        print(f"    [SPADE DEBUG] ✓ Patrón ensanchado (+0.20)")
        
        if grows_to_middle:
            score += 0.10
            print(f"    [SPADE DEBUG] ✓ Crece hacia middle (+0.10)")
        
        if shrinks_to_bottom:
            score += 0.05
            print(f"    [SPADE DEBUG] ✓ Decrece hacia bottom (+0.05)")
    
    if bottom_narrowest:
        score += 0.15  # Reducido de 0.20
        print(f"    [SPADE DEBUG] ✓ Bottom muy estrecho (+0.15)")
    
    if few_defects:
        score += 0.20
        print(f"    [SPADE DEBUG] ✓ Pocos defectos (+0.20)")
    
    if is_symmetric:
        score += 0.15
        print(f"    [SPADE DEBUG] ✓ Simétrica (+0.15)")
    
    # NUEVO: Bonus por solidez ALTA (característico de pica)
    if solidity > 0.85:
        score += 0.15
        print(f"    [SPADE DEBUG] ✓ Solidez alta (+0.15)")
    elif solidity > 0.82:
        score += 0.10
        print(f"    [SPADE DEBUG] ✓ Solidez ok (+0.10)")
    
    if good_aspect:
        score += 0.05
        print(f"    [SPADE DEBUG] ✓ Aspect ok (+0.05)")
    
    # Bonus patrón perfecto
    if top_narrower and max_width_in_middle and bottom_narrowest and few_defects and is_symmetric and solidity > 0.82:
        score += 0.05
        print(f"    [SPADE DEBUG] ✓ PATRÓN PERFECTO (+0.05)")
    
    print(f"    [SPADE DEBUG] SCORE FINAL: {score:.3f}")
    
    # Condición de aceptación MÁS ESTRICTA
    is_spade = (
        solidity > 0.82 and  # CRÍTICO: Alta solidez
        top_narrower and 
        max_width_in_middle and 
        few_defects and 
        is_symmetric and
        (bottom_narrowest or shrinks_to_bottom)
    )
    
    # O score MUY alto
    if score >= 0.80 and solidity > 0.80:  # Ambas condiciones
        is_spade = True
        print(f"    [SPADE DEBUG] ✓ PICA CONFIRMADA")
    
    if not is_spade:
        print(f"    [SPADE DEBUG] ✗ NO ES PICA")
    
    return is_spade, score

def detect_club_pattern(img):
    """
    Detecta trébol: BAJA solidez (múltiples lóbulos) + forma característica
    VERSION REFINADA: Usa SOLIDEZ BAJA como indicador principal
    """
    h, w = img.shape
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    
    if area < 100:
        return False, 0.0
    
    # 1. CRITERIO FUNDAMENTAL: Solidez BAJA (múltiples lóbulos)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Trébol tiene solidez BAJA (los 3 lóbulos crean huecos)
    low_solidity = solidity < 0.82  # Característico de trébol
    
    # 2. Defectos de convexidad (hendiduras entre lóbulos)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    
    try:
        defects = cv2.convexityDefects(cnt, hull_indices)
    except:
        defects = None
    
    num_defects = len(defects) if defects is not None else 0
    has_multiple_defects = num_defects >= 5  # Más estricto
    
    # 3. Análisis de TODA la imagen (no solo top)
    # El trébol puede estar orientado de cualquier forma
    
    # Calcular centroide
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2
    
    # Dividir en 4 cuadrantes desde el centro
    top_half = img[:cy, :]
    bottom_half = img[cy:, :]
    
    top_pixels = cv2.countNonZero(top_half)
    bottom_pixels = cv2.countNonZero(bottom_half)
    total_pixels = top_pixels + bottom_pixels
    
    # Trébol puede tener masa distribuida (no necesariamente top-heavy)
    balanced_mass = 0.35 < (top_pixels / total_pixels) < 0.75 if total_pixels > 0 else False
    
    # 4. Aspect ratio
    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
    
    # Trébol tiende a ser más redondeado
    roundish = 0.70 < aspect_ratio < 1.30
    
    # 5. Análisis de complejidad del contorno
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Trébol tiene circularidad baja (forma compleja)
    complex_shape = circularity < 0.75
    
    # Debug
    print(f"    [CLUB DEBUG] Solidez: {solidity:.3f} (req: <0.82)")
    print(f"    [CLUB DEBUG] Defectos: {num_defects} (req: >=5)")
    print(f"    [CLUB DEBUG] Circularidad: {circularity:.3f} (req: <0.75)")
    print(f"    [CLUB DEBUG] Aspect ratio: {aspect_ratio:.3f}")
    print(f"    [CLUB DEBUG] Masa balanceada: {balanced_mass}")
    print(f"    [CLUB DEBUG] Low solidity: {low_solidity}, Multiple defects: {has_multiple_defects}")
    print(f"    [CLUB DEBUG] Complex shape: {complex_shape}, Roundish: {roundish}")
    
    # Scoring
    score = 0.0
    
    # CRITERIO 1: Solidez BAJA (35% - FUNDAMENTAL)
    if low_solidity:
        score += 0.35
        print(f"    [CLUB DEBUG] ✓ Solidez baja (+0.35) - Múltiples lóbulos")
        
        # Bonus por solidez MUY baja
        if solidity < 0.75:
            score += 0.10
            print(f"    [CLUB DEBUG] ✓ Solidez muy baja (+0.10)")
    
    # CRITERIO 2: Múltiples defectos (25%)
    if has_multiple_defects:
        score += 0.25
        print(f"    [CLUB DEBUG] ✓ Múltiples defectos (+0.25)")
    elif num_defects >= 3:
        score += 0.15
        print(f"    [CLUB DEBUG] ~ Algunos defectos (+0.15)")
    
    # CRITERIO 3: Forma compleja (20%)
    if complex_shape:
        score += 0.20
        print(f"    [CLUB DEBUG] ✓ Forma compleja (+0.20)")
    
    # CRITERIO 4: Forma redondeada (10%)
    if roundish:
        score += 0.10
        print(f"    [CLUB DEBUG] ✓ Forma redondeada (+0.10)")
    
    # CRITERIO 5: Masa balanceada (10%)
    if balanced_mass:
        score += 0.10
        print(f"    [CLUB DEBUG] ✓ Masa balanceada (+0.10)")
    
    print(f"    [CLUB DEBUG] SCORE FINAL: {score:.3f}")
    
    # Condición de aceptación
    is_club = False
    
    # CAMINO 1: Solidez baja + defectos múltiples (característico)
    if low_solidity and has_multiple_defects:
        is_club = True
        print(f"    [CLUB DEBUG] ✓ TRÉBOL CONFIRMADO (solidez baja + defectos)")
    
    # CAMINO 2: Solidez baja + forma compleja
    elif low_solidity and complex_shape and num_defects >= 3:
        is_club = True
        print(f"    [CLUB DEBUG] ✓ TRÉBOL CONFIRMADO (solidez + complejidad)")
    
    # CAMINO 3: Score alto
    elif score >= 0.65:
        is_club = True
        print(f"    [CLUB DEBUG] ✓ TRÉBOL CONFIRMADO (score alto)")
    
    else:
        print(f"    [CLUB DEBUG] ✗ NO es trébol")
    
    return is_club, score

def classify_suit_by_pattern(img, color):
    """
    Clasifica el palo usando detección de patrones específicos
    VERSIÓN MEJORADA: Devuelve TODOS los scores, no solo el mejor
    """
    if color == 'red':
        # Detectar AMBOS palos rojos
        is_diamond, diamond_score = detect_diamond_pattern(img)
        is_heart, heart_score = detect_heart_pattern(img)
        
        print(f"  [PATTERN] Diamante: {is_diamond} (score: {diamond_score:.3f})")
        print(f"  [PATTERN] Corazón: {is_heart} (score: {heart_score:.3f})")
        
        # NUEVO: Retornar scores individuales
        scores = {
            'diamonds': diamond_score,
            'hearts': heart_score
        }
        
        # Determinar el mejor
        if diamond_score > heart_score:
            best = 'diamonds'
        else:
            best = 'hearts'
        
        return best, scores
            
    else:  # black
        # Detectar AMBOS palos negros
        is_spade, spade_score = detect_spade_pattern(img)
        is_club, club_score = detect_club_pattern(img)
        
        print(f"  [PATTERN] Pica: {is_spade} (score: {spade_score:.3f})")
        print(f"  [PATTERN] Trébol: {is_club} (score: {club_score:.3f})")
        
        # NUEVO: Retornar scores individuales
        scores = {
            'spades': spade_score,
            'clubs': club_score
        }
        
        # Determinar el mejor
        if club_score > spade_score:
            best = 'clubs'
        else:
            best = 'spades'
        
        return best, scores

def match_template_multi_method(img, template):
    """
    Comparación por template matching
    """
    scores = []
    
    res1 = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    scores.append(res1.max())
    
    res2 = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    scores.append(res2.max())
    
    contours_img, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_tmpl, _ = cv2.findContours(template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_img and contours_tmpl:
        cnt_img = max(contours_img, key=cv2.contourArea)
        cnt_tmpl = max(contours_tmpl, key=cv2.contourArea)
        
        match_distance = cv2.matchShapes(cnt_img, cnt_tmpl, cv2.CONTOURS_MATCH_I2, 0)
        match_similarity = 1.0 / (1.0 + match_distance * 5)
        scores.append(match_similarity)
    
    return np.mean(scores) if scores else 0.0

def match_suit_by_color(img, color):
    """
    Combina detección de patrones con template matching
    VERSION MEJORADA: Mejor resolución de conflictos spade vs club
    """
    # PASO 1: Detección por patrones geométricos
    best_by_pattern, pattern_scores = classify_suit_by_pattern(img, color)
    
    print(f"  Clasificación por patrón: {best_by_pattern}")
    print(f"  Scores de patrón: {pattern_scores}")
    
    # PASO 2: Template matching con MÚLTIPLES VARIANTES
    color_templates = SUIT_TEMPLATES.get(color, {})
    template_scores = {}
    
    # Comparar contra PRIMARY templates
    for suit_name, template_list in color_templates.get('primary', {}).items():
        max_score = 0.0
        for template in template_list:
            score = match_template_multi_method(img, template)
            max_score = max(max_score, score)
        
        if suit_name not in template_scores:
            template_scores[suit_name] = {'primary': [], 'secondary': []}
        
        template_scores[suit_name]['primary'].append(max_score)
    
    # Comparar contra SECONDARY templates (si existen)
    if len(color_templates.get('secondary', {})) > 0:
        for suit_name, template_list in color_templates.get('secondary', {}).items():
            max_score = 0.0
            for template in template_list:
                score = match_template_multi_method(img, template)
                max_score = max(max_score, score)
            
            if suit_name not in template_scores:
                template_scores[suit_name] = {'primary': [], 'secondary': []}
            
            template_scores[suit_name]['secondary'].append(max_score)
    
    # Calcular scores finales (promedio ponderado)
    final_template_scores = {}
    
    for suit_name, scores_dict in template_scores.items():
        primary_scores = scores_dict.get('primary', [])
        secondary_scores = scores_dict.get('secondary', [])
        
        primary_avg = max(primary_scores) if primary_scores else 0.0
        secondary_avg = max(secondary_scores) if secondary_scores else 0.0
        
        # Ponderación: Primary 50%, Secondary 50% (equilibrado para suits)
        if secondary_avg > 0:
            final_score = 0.50 * primary_avg + 0.50 * secondary_avg
        else:
            final_score = primary_avg
        
        final_template_scores[suit_name] = final_score
    
    print(f"  Scores de template: {final_template_scores}")
    
    # PASO 3: Combinar scores
    final_scores = {}
    possible_suits = ['diamonds', 'hearts'] if color == 'red' else ['spades', 'clubs']
    
    # DETECCIÓN DE CONFLICTOS EN PALOS NEGROS
    if color == 'black':
        spade_score = pattern_scores.get('spades', 0.0)
        club_score = pattern_scores.get('clubs', 0.0)
        
        # CASO 1: SPADE detectado con alta confianza (>= 0.80)
        if spade_score >= 0.80:
            print(f"  [PRIORIDAD] Spade score muy alto ({spade_score:.3f}) → FORZAR SPADE")
            
            # FORZAR SPADE (ignorar club)
            final_scores['spades'] = spade_score
            final_scores['clubs'] = 0.0
            
            print(f"  [spades] Final: {spade_score:.3f} (FORZADO)")
            print(f"  [clubs] Final: 0.000 (DESCARTADO)")
            
            return 'spades', spade_score, final_scores
        
        # CASO 2: CLUB detectado con alta confianza Y spade bajo
        if club_score >= 0.85 and spade_score < 0.50:
            print(f"  [PRIORIDAD] Club score muy alto ({club_score:.3f}) y spade bajo → FORZAR CLUB")
            
            final_scores['clubs'] = club_score
            final_scores['spades'] = 0.0
            
            print(f"  [clubs] Final: {club_score:.3f} (FORZADO)")
            print(f"  [spades] Final: 0.000 (DESCARTADO)")
            
            return 'clubs', club_score, final_scores
        
        # CASO 3: Conflicto (ambos altos)
        if spade_score >= 0.70 and club_score >= 0.70:
            print(f"  [CONFLICTO] Ambos palos altos!")
            print(f"    Spade: {spade_score:.3f}")
            print(f"    Club: {club_score:.3f}")
            
            # REGLA DE DESEMPATE: Diferencia significativa (>0.15)
            diff = abs(spade_score - club_score)
            
            if diff > 0.15:
                # Hay un ganador claro
                if spade_score > club_score:
                    print(f"  [RESOLUCIÓN] Spade gana por diferencia ({diff:.3f})")
                    final_scores['spades'] = spade_score
                    final_scores['clubs'] = 0.0
                    return 'spades', spade_score, final_scores
                else:
                    print(f"  [RESOLUCIÓN] Club gana por diferencia ({diff:.3f})")
                    final_scores['clubs'] = club_score
                    final_scores['spades'] = 0.0
                    return 'clubs', club_score, final_scores
            else:
                # Muy cerrado: usar template matching como tiebreaker
                print(f"  [RESOLUCIÓN] Diferencia pequeña, usar templates...")
                spade_tmpl = final_template_scores.get('spades', 0.0)  # ← CORRECTO
                club_tmpl = final_template_scores.get('clubs', 0.0)    # ← CORRECTO
                
                if spade_tmpl > club_tmpl:
                    print(f"  [TIEBREAK] Spade gana (template: {spade_tmpl:.3f} vs {club_tmpl:.3f})")
                    final_scores['spades'] = (spade_score + spade_tmpl) / 2
                    final_scores['clubs'] = 0.0
                    return 'spades', final_scores['spades'], final_scores
                else:
                    print(f"  [TIEBREAK] Club gana (template: {club_tmpl:.3f} vs {spade_tmpl:.3f})")
                    final_scores['clubs'] = (club_score + club_tmpl) / 2
                    final_scores['spades'] = 0.0
                    return 'clubs', final_scores['clubs'], final_scores
        
        # Sin conflicto: pesos normales
        weight_pattern = 0.90
        weight_template = 0.10
    else:
        # Rojos: pesos normales siempre
        weight_pattern = 0.90
        weight_template = 0.10
    
    # Calcular scores finales (sin conflicto)
    for suit in possible_suits:
        pattern_score = pattern_scores.get(suit, 0.0)
        template_score = final_template_scores.get(suit, 0.0)
        
        final_score = weight_pattern * pattern_score + weight_template * template_score
        final_scores[suit] = final_score
        
        print(f"  [{suit}] Final: {final_score:.3f} (pattern: {pattern_score:.3f}, template: {template_score:.3f})")
    
    # Elegir el mejor
    if final_scores:
        best_suit = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_suit]
        return best_suit, best_score, final_scores
    
    return best_by_pattern, pattern_scores.get(best_by_pattern, 0.0), pattern_scores

def extract_suit_roi_dynamic(warp):
    """
    Extrae el ROI del símbolo de palo de forma DINÁMICA
    VERSION MEJORADA: Maneja mejor cartas de figuras (J, Q, K)
    """
    h, w = warp.shape[:2]
    
    # Definir zona de búsqueda
    search_h = int(h * 0.60)
    search_w = int(w * 0.40)
    search_region = warp[0:search_h, 0:search_w]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
    
    # Binarizar
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Limpieza mínima
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Buscar contornos en la zona de búsqueda
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("  [ROI DINÁMICO] ⚠ No se encontraron contornos, usando ROI fijo por defecto")
        return warp[100:170, 10:60], (10, 100, 60, 170)
    
    # Filtrar contornos válidos
    min_area = 200
    max_area = 3000
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect = float(w_box) / h_box if h_box > 0 else 0
            
            # Aceptar suits tanto verticales como horizontales
            if 0.4 < aspect < 2.5:  # Más permisivo
                valid_contours.append(cnt)
    
    if not valid_contours:
        print("  [ROI DINÁMICO] ⚠ No hay contornos válidos, usando ROI fijo")
        return warp[100:170, 10:60], (10, 100, 60, 170)
    
    # NUEVA ESTRATEGIA: Scoring multi-criterio
    candidates = []
    
    for cnt in valid_contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        center_y = y + h_box // 2
        center_x = x + w_box // 2
        aspect = float(w_box) / h_box if h_box > 0 else 0
        
        score = 0.0
        
        # 1. Posición vertical (30% weight)
        rel_pos = center_y / search_h
        if 0.30 < rel_pos < 0.65:
            score += 30 * (1 - abs(rel_pos - 0.45) / 0.15)
        
        # 2. Posición horizontal (20% weight)
        rel_pos_x = center_x / search_w
        if 0.10 < rel_pos_x < 0.50:
            score += 20 * (1 - abs(rel_pos_x - 0.25) / 0.25)
        
        # 3. Área razonable (25% weight)
        if 400 < area < 1500:
            optimal_area = 800
            score += 25 * (1 - abs(area - optimal_area) / optimal_area)
        
        # 4. Aspect ratio (25% weight)
        # Preferir vertical (0.6-1.3)
        if 0.6 < aspect < 1.3:
            score += 25 * (1 - abs(aspect - 0.85) / 0.45)
        
        candidates.append({
            'contour': cnt,
            'x': x,
            'y': y,
            'w': w_box,
            'h': h_box,
            'area': area,
            'aspect': aspect,
            'score': score
        })
        
        print(f"  [ROI DINÁMICO] Candidato: pos=({x},{y}), size=({w_box}x{h_box}), area={area:.0f}, aspect={aspect:.2f}, score={score:.1f}")
    
    if not candidates:
        print("  [ROI DINÁMICO] ⚠ Sin candidatos válidos, usando ROI fijo")
        return warp[100:170, 10:60], (10, 100, 60, 170)
    
    # Seleccionar el candidato con mayor score
    candidates.sort(key=lambda c: c['score'], reverse=True)
    selected = candidates[0]
    
    x = selected['x']
    y = selected['y']
    w_box = selected['w']
    h_box = selected['h']
    
    print(f"  [ROI DINÁMICO] ✓ Contorno seleccionado: pos=({x},{y}), size=({w_box}x{h_box}), score={selected['score']:.1f}")
    
    # NUEVO: Expandir MÁS para capturar símbolo completo
    expansion = 0.25  # 25% (antes 20%)
    x_exp = max(0, int(x - w_box * expansion))
    y_exp = max(0, int(y - h_box * expansion))
    w_exp = min(search_w, int(x + w_box * (1 + expansion)))
    h_exp = min(search_h, int(y + h_box * (1 + expansion)))
    
    # Asegurar dimensiones mínimas MÁS GRANDES
    roi_w = w_exp - x_exp
    roi_h = h_exp - y_exp
    
    # Forzar tamaño mínimo 60x70 (antes 50x60)
    min_w = 60
    min_h = 70
    
    if roi_w < min_w or roi_h < min_h:
        print(f"  [ROI DINÁMICO] ⚠ ROI pequeño ({roi_w}x{roi_h}), expandiendo a mínimo {min_w}x{min_h}...")
        
        center_x = x + w_box // 2
        center_y = y + h_box // 2
        
        x_exp = max(0, center_x - min_w // 2)
        y_exp = max(0, center_y - min_h // 2)
        w_exp = min(search_w, x_exp + min_w)
        h_exp = min(search_h, y_exp + min_h)
        
        # Ajustar si se sale de límites
        if w_exp > search_w:
            x_exp = max(0, search_w - min_w)
            w_exp = search_w
        if h_exp > search_h:
            y_exp = max(0, search_h - min_h)
            h_exp = search_h
    
    # Extraer ROI
    roi = search_region[y_exp:h_exp, x_exp:w_exp]
    
    print(f"  [ROI DINÁMICO] ✓ ROI final: [{y_exp}:{h_exp}, {x_exp}:{w_exp}] = {roi.shape}")
    
    return roi, (x_exp, y_exp, w_exp, h_exp)

def recognize_suit(warp):
    """
    Reconoce el palo de una carta desde su imagen warpeada
    VERSION MEJORADA: Selecciona automáticamente la MEJOR rotación
    """
    print(f"\n{'='*60}")
    print(f"INICIANDO RECONOCIMIENTO DE PALO")
    print(f"{'='*60}")
    
    cv2.imwrite("debug_suit_warp_full.png", warp)
    
    # Extraer ROI de forma DINÁMICA
    suit_roi, roi_coords = extract_suit_roi_dynamic(warp)
    x1, y1, x2, y2 = roi_coords
    
    cv2.imwrite("debug_suit_roi_original.png", suit_roi)
    
    # Debug: Mostrar ubicación del ROI
    debug_warp = warp.copy()
    cv2.rectangle(debug_warp, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(debug_warp, 'SUIT ROI', (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite("debug_suit_roi_location.png", debug_warp)
    
    print(f"\nTamaño carta warpeada: {warp.shape}")
    print(f"ROI extraído: [{y1}:{y2}, {x1}:{x2}] = {suit_roi.shape}")
    
    # Determinar color
    is_red = is_red_suit(suit_roi)
    color = 'red' if is_red else 'black'
    print(f"\n{'*'*60}")
    print(f"COLOR DETECTADO: {color.upper()}")
    print(f"{'*'*60}")
    
    # Preprocesar
    suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_suit_01_gray.png", suit_gray)
    
    _, suit_bin = cv2.threshold(suit_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("debug_suit_02_binary.png", suit_bin)
    
    # Morfología mínima
    kernel = np.ones((2, 2), np.uint8)
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite("debug_suit_03_open.png", suit_bin)
    
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite("debug_suit_04_close.png", suit_bin)
    
    # Extraer contorno principal
    contours, _ = cv2.findContours(suit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(suit_bin)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        suit_bin = cv2.bitwise_and(suit_bin, mask)
    
    cv2.imwrite("debug_suit_05_contour.png", suit_bin)
    
    # Resize manteniendo aspect ratio
    h_roi, w_roi = suit_bin.shape
    target_size = 70
    
    if h_roi > w_roi:
        scale = target_size / h_roi
        new_h = target_size
        new_w = int(w_roi * scale)
    else:
        scale = target_size / w_roi
        new_w = target_size
        new_h = int(h_roi * scale)
    
    suit_bin = cv2.resize(suit_bin, (new_w, new_h))
    
    # Centrar en canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = suit_bin
    suit_bin = canvas
    
    cv2.imwrite("debug_suit_06_resized.png", suit_bin)
    
    # Reconocimiento con rotaciones
    best_suit_name = None
    best_suit_score = -1
    best_angle = 0
    all_rotations_results = {}
    
    for angle in [0, 90, 180, 270]:
        print(f"\n{'='*60}")
        print(f"ROTACIÓN {angle}°")
        print(f"{'='*60}")
        
        if angle > 0:
            M = cv2.getRotationMatrix2D((35, 35), angle, 1.0)
            suit_rotated = cv2.warpAffine(suit_bin, M, (70, 70))
        else:
            suit_rotated = suit_bin
        
        cv2.imwrite(f"debug_suit_rotation_{angle}.png", suit_rotated)
        
        suit_name, suit_score, scores_dict = match_suit_by_color(suit_rotated, color)
        
        all_rotations_results[angle] = {
            'name': suit_name,
            'score': suit_score,
            'all_scores': scores_dict.copy()
        }
        
        print(f"\n>>> RESULTADO ROTACIÓN {angle}°: {suit_name} (score: {suit_score:.3f})")
        
        if suit_score > best_suit_score:
            best_suit_score = suit_score
            best_suit_name = suit_name
            best_angle = angle
    
    # NUEVA LÓGICA: Si hay EMPATE o confusión, usar criterios adicionales
    if color == 'black':
        # Contar cuántas rotaciones votaron por cada palo
        spade_votes = sum(1 for r in all_rotations_results.values() if r['name'] == 'spades')
        club_votes = sum(1 for r in all_rotations_results.values() if r['name'] == 'clubs')
        
        # Calcular promedio de scores (ignorando rotaciones con score 0)
        spade_scores = [r['all_scores'].get('spades', 0) for r in all_rotations_results.values() if r['all_scores'].get('spades', 0) > 0]
        club_scores = [r['all_scores'].get('clubs', 0) for r in all_rotations_results.values() if r['all_scores'].get('clubs', 0) > 0]
        
        avg_spade = sum(spade_scores) / len(spade_scores) if spade_scores else 0
        avg_club = sum(club_scores) / len(club_scores) if club_scores else 0
        
        print(f"\n{'='*60}")
        print(f"ANÁLISIS DE CONSISTENCIA:")
        print(f"{'='*60}")
        print(f"  Votos spades: {spade_votes}/4, promedio score: {avg_spade:.3f}")
        print(f"  Votos clubs: {club_votes}/4, promedio score: {avg_club:.3f}")
        
        # Si una tiene MÁS votos consistentes (2+ rotaciones)
        if spade_votes >= 2 and club_votes <= 1:
            print(f"  [DECISIÓN] Spades tiene mayoría de votos → SPADES")
            best_suit_name = 'spades'
            best_suit_score = avg_spade
        elif club_votes >= 3:  # Club necesita 3+ votos (más estricto)
            print(f"  [DECISIÓN] Clubs tiene mayoría absoluta → CLUBS")
            best_suit_name = 'clubs'
            best_suit_score = avg_club
        elif spade_votes == club_votes:
            # Empate: usar el de mayor score promedio
            if avg_spade > avg_club * 1.1:  # 10% mejor
                print(f"  [DESEMPATE] Spades tiene mejor score promedio → SPADES")
                best_suit_name = 'spades'
                best_suit_score = avg_spade
            elif avg_club > avg_spade * 1.1:
                print(f"  [DESEMPATE] Clubs tiene mejor score promedio → CLUBS")
                best_suit_name = 'clubs'
                best_suit_score = avg_club
    
    # RESUMEN FINAL
    print(f"\n{'='*60}")
    print(f"RESUMEN DE TODAS LAS ROTACIONES:")
    print(f"{'='*60}")
    for angle, result in all_rotations_results.items():
        marker = " ← ELEGIDO" if angle == best_angle else ""
        print(f"\nRotación {angle}°: {result['name']} (score: {result['score']:.3f}){marker}")
        print(f"  Scores detallados: {result['all_scores']}")
    
    print(f"\n{'*'*60}")
    print(f"*** DECISIÓN FINAL ***")
    print(f"PALO: {best_suit_name}")
    print(f"ROTACIÓN: {best_angle}°")
    print(f"SCORE: {best_suit_score:.3f}")
    print(f"{'*'*60}\n")
    
    return best_suit_name, best_suit_score