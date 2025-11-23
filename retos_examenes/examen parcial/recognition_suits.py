import cv2
import numpy as np
import os

def load_suit_templates(path='templates/suits', size=(70,70)):
    """Carga las plantillas de los palos organizadas por color"""
    templates = {
        'red': {},
        'black': {}
    }
    
    for name in os.listdir(path):
        if not name.lower().endswith('.png'):
            continue
        
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img = cv2.resize(img, size)
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Limpiar y normalizar la plantilla
        kernel = np.ones((3,3), np.uint8)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Clasificar por color basándose en el nombre
        base_name = os.path.splitext(name)[0].lower()
        if any(word in base_name for word in ['hearts', 'diamonds', 'corazones', 'diamantes', 'oros']):
            templates['red'][base_name] = img_bin
        elif any(word in base_name for word in ['spades', 'clubs', 'picas', 'treboles', 'espadas', 'bastos']):
            templates['black'][base_name] = img_bin
    
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
    Detecta pica: PUNTA ESTRECHA Y CENTRADA arriba
    """
    h, w = img.shape
    
    # Analizar defectos
    defects_count, max_depth, _ = analyze_convexity_defects(img)
    moderate_defects = 2 <= defects_count <= 5
    
    # Solidez media
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area = cv2.contourArea(cnt)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        medium_solidity = 0.75 < solidity < 0.90  # Más permisivo
    else:
        medium_solidity = False
        solidity = 0
    
    # CARACTERÍSTICA CLAVE: Top ESTRECHO (punta única)
    top_10 = img[:int(h*0.1), :]
    top_projection = np.sum(top_10, axis=0)
    
    narrow_top = False
    centered_peak = False
    top_width_ratio = 1.0
    
    if len(top_projection) > 0 and np.max(top_projection) > 0:
        top_projection = top_projection / np.max(top_projection)
        high_intensity_cols = np.sum(top_projection > 0.5)
        total_cols = len(top_projection)
        top_width_ratio = high_intensity_cols / total_cols if total_cols > 0 else 1.0
        
        # Pica: top MUY estrecho
        narrow_top = top_width_ratio < 0.35
        
        # Verificar que la punta está centrada
        center = len(top_projection) // 2
        center_range = range(max(0, center-10), min(len(top_projection), center+10))
        center_mass = np.sum([top_projection[i] for i in center_range if i < len(top_projection)])
        total_mass = np.sum(top_projection)
        centered_peak = (center_mass / total_mass) > 0.35 if total_mass > 0 else False
    
    # Debug
    print(f"    [SPADE DEBUG] Defectos: {defects_count}, Solidez: {solidity:.3f}")
    print(f"    [SPADE DEBUG] Top width ratio: {top_width_ratio:.3f} (req: <0.35)")
    print(f"    [SPADE DEBUG] Top estrecho: {narrow_top}, Centrado: {centered_peak}")
    
    # Score - PESO MÁXIMO al top estrecho
    score = 0.0
    
    # CRITERIO FUNDAMENTAL: Top estrecho + centrado
    if narrow_top and centered_peak:
        score += 0.70  # PESO ENORME para la característica distintiva
        
        if medium_solidity:
            score += 0.15
        
        if moderate_defects:
            score += 0.10
        
        if top_width_ratio < 0.30:  # Muy muy estrecho
            score += 0.05
        
        print(f"    [SPADE DEBUG] ✓ Punta detectada, score: {score:.3f}")
    
    # Penalizar si el top NO es estrecho
    elif not narrow_top:
        score = 0.0
        print(f"    [SPADE DEBUG] ✗ RECHAZADO: Top NO es estrecho ({top_width_ratio:.3f} >= 0.35)")
    
    # REQUIERE ambos: estrecho Y centrado
    is_spade = (narrow_top and centered_peak)
    
    return is_spade, score

def detect_club_pattern(img):
    """
    Detecta trébol: 3 CÍRCULOS arriba + tallo abajo
    VERSIÓN MEJORADA para detectar los 3 lóbulos
    """
    h, w = img.shape
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    
    if area < 100:
        return False, 0.0
    
    # 1. Defectos de convexidad (hendiduras entre los 3 círculos)
    hull = cv2.convexHull(cnt, returnPoints=False)
    
    try:
        defects = cv2.convexityDefects(cnt, hull)
    except:
        defects = None
    
    num_defects = len(defects) if defects is not None else 0
    
    # Trébol debe tener MÚLTIPLES defectos (entre los 3 lóbulos)
    has_multiple_defects = num_defects >= 3
    
    # 2. Solidez (área vs hull)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Trébol tiene solidez alta (forma compacta)
    high_solidity = solidity > 0.75
    
    # 3. Análisis del TOP (donde están los 3 círculos)
    top_40_percent = int(h * 0.40)
    top_region = img[:top_40_percent, :]
    
    # Proyección horizontal del top
    top_projection = np.sum(top_region, axis=0)
    
    if len(top_projection) > 0 and np.max(top_projection) > 0:
        # Normalizar
        top_projection = top_projection.astype(float) / np.max(top_projection)
        
        # Buscar PICOS en la proyección (los 3 lóbulos)
        # Un pico es una columna con valor alto rodeada de valores más bajos
        peaks = []
        for i in range(2, len(top_projection) - 2):
            if top_projection[i] > 0.5:  # Umbral de intensidad
                # Verificar si es un pico local
                is_peak = (top_projection[i] >= top_projection[i-1] and 
                          top_projection[i] >= top_projection[i-2] and
                          top_projection[i] >= top_projection[i+1] and
                          top_projection[i] >= top_projection[i+2])
                
                # Evitar picos muy cercanos (mismo lóbulo)
                if is_peak:
                    if not peaks or (i - peaks[-1] > w * 0.15):
                        peaks.append(i)
        
        num_peaks = len(peaks)
        
        # Ancho del top ocupado
        high_intensity_cols = np.sum(top_projection > 0.3)
        top_width_ratio = high_intensity_cols / w if w > 0 else 0
        
        # Trébol tiene top ANCHO (los 3 círculos ocupan casi todo el ancho)
        wide_top = top_width_ratio > 0.60
    else:
        num_peaks = 0
        top_width_ratio = 0.0
        wide_top = False
    
    # 4. Análisis del BOTTOM (tallo)
    bottom_25_percent = int(h * 0.25)
    bottom_region = img[h - bottom_25_percent:, :]
    
    # Proyección horizontal del bottom
    bottom_projection = np.sum(bottom_region, axis=0)
    
    if len(bottom_projection) > 0 and np.max(bottom_projection) > 0:
        bottom_projection = bottom_projection.astype(float) / np.max(bottom_projection)
        
        # Ancho del bottom ocupado
        high_intensity_cols = np.sum(bottom_projection > 0.3)
        bottom_width_ratio = high_intensity_cols / w if w > 0 else 0
        
        # Tallo debe ser ESTRECHO
        narrow_bottom = bottom_width_ratio < 0.40
    else:
        bottom_width_ratio = 0.0
        narrow_bottom = False
    
    # 5. Distribución de masa (top-heavy)
    top_70_region = img[:int(h * 0.70), :]
    bottom_30_region = img[int(h * 0.70):, :]
    
    top_pixels = cv2.countNonZero(top_70_region)
    bottom_pixels = cv2.countNonZero(bottom_30_region)
    total_pixels = top_pixels + bottom_pixels
    
    top_mass_ratio = top_pixels / total_pixels if total_pixels > 0 else 0
    top_heavy = top_mass_ratio > 0.60
    
    # Debug detallado
    print(f"    [CLUB DEBUG] Defectos: {num_defects}, Solidez: {solidity:.3f}")
    print(f"    [CLUB DEBUG] Picos detectados en top: {num_peaks}")
    print(f"    [CLUB DEBUG] Top width ratio: {top_width_ratio:.3f} (req: >0.60)")
    print(f"    [CLUB DEBUG] Bottom width ratio: {bottom_width_ratio:.3f} (req: <0.40)")
    print(f"    [CLUB DEBUG] Top mass ratio: {top_mass_ratio:.3f} (req: >0.60)")
    print(f"    [CLUB DEBUG] Wide top: {wide_top}, Narrow bottom: {narrow_bottom}")
    print(f"    [CLUB DEBUG] Multiple defects: {has_multiple_defects}, Top heavy: {top_heavy}")
    
    # SCORE - Sistema de puntos
    score = 0.0
    
    # CRITERIO PRINCIPAL: 3 picos detectados (característica única del trébol)
    if num_peaks >= 3:
        score += 0.40
        print(f"    [CLUB DEBUG] ✓ 3 picos detectados (+0.40)")
    elif num_peaks == 2:
        score += 0.20
        print(f"    [CLUB DEBUG] ~ 2 picos detectados (+0.20)")
    
    # CRITERIO FUERTE: Top ancho
    if wide_top:
        score += 0.25
        print(f"    [CLUB DEBUG] ✓ Top ancho (+0.25)")
    elif top_width_ratio > 0.50:
        score += 0.15
        print(f"    [CLUB DEBUG] ~ Top medio (+0.15)")
    
    # CRITERIO FUERTE: Múltiples defectos
    if has_multiple_defects:
        score += 0.20
        print(f"    [CLUB DEBUG] ✓ Múltiples defectos (+0.20)")
    elif num_defects >= 2:
        score += 0.10
        print(f"    [CLUB DEBUG] ~ Algunos defectos (+0.10)")
    
    # CRITERIO MEDIO: Bottom estrecho
    if narrow_bottom:
        score += 0.10
        print(f"    [CLUB DEBUG] ✓ Bottom estrecho (+0.10)")
    
    # CRITERIO MENOR: Top heavy
    if top_heavy:
        score += 0.05
        print(f"    [CLUB DEBUG] ✓ Top heavy (+0.05)")
    
    print(f"    [CLUB DEBUG] SCORE FINAL: {score:.3f}")
    
    # Condiciones de aceptación
    is_club = False
    
    # CAMINO 1: 3 picos + características adicionales
    if num_peaks >= 3 and (wide_top or has_multiple_defects):
        is_club = True
        print(f"    [CLUB DEBUG] ✓ TRÉBOL CONFIRMADO (3 picos)")
    
    # CAMINO 2: Top ancho + múltiples defectos + características adicionales
    elif wide_top and has_multiple_defects and (narrow_bottom or top_heavy):
        is_club = True
        print(f"    [CLUB DEBUG] ✓ TRÉBOL CONFIRMADO (forma completa)")
    
    # CAMINO 3: Score alto suficiente
    elif score >= 0.70:
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
    VERSIÓN MEJORADA: Resuelve conflictos spade vs club
    """
    # PASO 1: Detección por patrones geométricos
    best_by_pattern, pattern_scores = classify_suit_by_pattern(img, color)
    
    print(f"  Clasificación por patrón: {best_by_pattern}")
    print(f"  Scores de patrón: {pattern_scores}")
    
    # PASO 2: Template matching
    templates = SUIT_TEMPLATES.get(color, {})
    template_scores = {}
    
    for name, tmpl in templates.items():
        score = match_template_multi_method(img, tmpl)
        base_name = name.split('_')[0]
        if base_name not in template_scores or score > template_scores[base_name]:
            template_scores[base_name] = score
    
    print(f"  Scores de template: {template_scores}")
    
    # PASO 3: Combinar scores
    final_scores = {}
    possible_suits = ['diamonds', 'hearts'] if color == 'red' else ['spades', 'clubs']
    
    # NUEVO: Detectar conflicto en palos negros
    if color == 'black':
        spade_detected = pattern_scores.get('spades', 0.0) > 0.5
        club_detected = pattern_scores.get('clubs', 0.0) > 0.5
        
        # CONFLICTO: Ambos detectados
        if spade_detected and club_detected:
            print(f"  [CONFLICTO] Ambos palos detectados!")
            print(f"    Spade score: {pattern_scores['spades']:.3f}")
            print(f"    Club score: {pattern_scores['clubs']:.3f}")
            
            # REGLA: Si club >= 0.85 (3 picos detectados), PRIORIZAR CLUB
            if pattern_scores['clubs'] >= 0.85:
                print(f"  [RESOLUCIÓN] Club score muy alto (>=0.85) → FORZAR CLUB")
                final_scores['clubs'] = pattern_scores['clubs']
                final_scores['spades'] = 0.0
                
                best_suit = 'clubs'
                best_score = final_scores['clubs']
                
                print(f"  [clubs] Final: {best_score:.3f} (FORZADO)")
                print(f"  [spades] Final: 0.000 (DESCARTADO)")
                
                return best_suit, best_score, final_scores
            
            # Si no es claro, usar pesos 50-50
            print(f"  [RESOLUCIÓN] Usar pesos equilibrados (50-50)")
            weight_pattern = 0.50
            weight_template = 0.50
        else:
            # Sin conflicto: pesos normales
            weight_pattern = 0.90
            weight_template = 0.10
    else:
        # Rojos: pesos normales siempre
        weight_pattern = 0.90
        weight_template = 0.10
    
    # Calcular scores finales
    for suit in possible_suits:
        pattern_score = pattern_scores.get(suit, 0.0)
        template_score = template_scores.get(suit, 0.0)
        
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
    VERSION MEJORADA: Detecta mejor en cartas de figuras (J, Q, K)
    """
    h, w = warp.shape[:2]
    
    # Definir zona de búsqueda (AMPLIADA para cartas de figuras)
    search_h = int(h * 0.60)  # 60% superior (antes 50%)
    search_w = int(w * 0.40)  # 40% izquierdo (antes 35%)
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
    min_area = 200  # Área mínima
    max_area = 3000  # NUEVO: Área máxima (evita capturar figuras grandes)
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            aspect = float(w_box) / h_box if h_box > 0 else 0
            
            # NUEVO: Filtrar por aspect ratio razonable
            # Suits tienen aspect 0.5-1.5 (ni muy anchos ni muy altos)
            if 0.4 < aspect < 2.0:
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
        
        # Calcular score basado en múltiples criterios
        score = 0.0
        
        # 1. Posición vertical (30% weight)
        # Preferir contornos en el rango [30%-60%] de altura
        rel_pos = center_y / search_h
        if 0.30 < rel_pos < 0.65:
            score += 30 * (1 - abs(rel_pos - 0.45) / 0.15)
        
        # 2. Posición horizontal (20% weight)
        # Preferir contornos cerca del borde izquierdo (10-30%)
        rel_pos_x = center_x / search_w
        if 0.10 < rel_pos_x < 0.50:
            score += 20 * (1 - abs(rel_pos_x - 0.25) / 0.25)
        
        # 3. Área razonable (25% weight)
        # Preferir áreas entre 400-1500
        if 400 < area < 1500:
            optimal_area = 800
            score += 25 * (1 - abs(area - optimal_area) / optimal_area)
        
        # 4. Aspect ratio (25% weight)
        # Preferir aspect ratio entre 0.7-1.3 (cercano a cuadrado)
        if 0.6 < aspect < 1.4:
            score += 25 * (1 - abs(aspect - 1.0) / 0.4)
        
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
    
    # Expandir el bounding box un 20% (antes 15%)
    expansion = 0.20
    x_exp = max(0, int(x - w_box * expansion))
    y_exp = max(0, int(y - h_box * expansion))
    w_exp = min(search_w, int(x + w_box * (1 + expansion)))
    h_exp = min(search_h, int(y + h_box * (1 + expansion)))
    
    # Asegurar dimensiones mínimas más estrictas
    roi_w = w_exp - x_exp
    roi_h = h_exp - y_exp
    
    # Forzar tamaño mínimo 50x60 (no 40x50)
    min_w = 50
    min_h = 60
    
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
            x_exp = search_w - min_w
            w_exp = search_w
        if h_exp > search_h:
            y_exp = search_h - min_h
            h_exp = search_h
    
    # Extraer ROI
    roi = search_region[y_exp:h_exp, x_exp:w_exp]
    
    print(f"  [ROI DINÁMICO] ✓ ROI final: [{y_exp}:{h_exp}, {x_exp}:{w_exp}] = {roi.shape}")
    
    return roi, (x_exp, y_exp, w_exp, h_exp)

def recognize_suit(warp):
    """
    Reconoce el palo de una carta desde su imagen warpeada
    VERSION DEBUG COMPLETA
    """
    print(f"\n{'='*60}")
    print(f"INICIANDO RECONOCIMIENTO DE PALO")
    print(f"{'='*60}")
    
    cv2.imwrite("debug_suit_warp_full.png", warp)
    
    suit_roi, roi_coords = extract_suit_roi_dynamic(warp)
    x1, y1, x2, y2 = roi_coords
    cv2.imwrite("debug_suit_roi_original.png", suit_roi)
    
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
    
    # Preprocesar - REDUCIDO
    suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("debug_suit_01_gray.png", suit_gray)
    
    _, suit_bin = cv2.threshold(suit_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("debug_suit_02_binary.png", suit_bin)
    
    # REDUCIR morfología para preservar esquinas
    kernel = np.ones((2,2), np.uint8)  # Kernel más pequeño (antes 3x3)
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_OPEN, kernel, iterations=1)  # Reducido de 2 a 1
    cv2.imwrite("debug_suit_03_open.png", suit_bin)
    
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_CLOSE, kernel, iterations=1)  # Reducido de 3 a 1
    cv2.imwrite("debug_suit_04_close.png", suit_bin)
    
    # Extraer contorno principal
    contours, _ = cv2.findContours(suit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(suit_bin)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        suit_bin = cv2.bitwise_and(suit_bin, mask)
    
    cv2.imwrite("debug_suit_05_contour.png", suit_bin)
    
    suit_bin = cv2.resize(suit_bin, (70, 70))
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