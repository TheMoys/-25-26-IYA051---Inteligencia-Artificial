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
    Detecta específicamente la hendidura superior característica del corazón
    """
    h, w = img.shape
    
    # Analizar solo el 20% superior (más restrictivo)
    top_section = img[:int(h*0.2), :]
    
    # Buscar la hendidura: región blanca rodeada de negro
    # Invertir para que el fondo sea blanco
    top_inverted = cv2.bitwise_not(top_section)
    
    # Encontrar componentes conectados (regiones blancas)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(top_inverted, connectivity=8)
    
    # Buscar una región blanca en el centro superior (la hendidura)
    notch_found = False
    notch_depth = 0
    
    for i in range(1, num_labels):  # Ignorar fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Debe estar en la parte central superior MUY ESTRICTO
        center_x = x + width / 2
        is_centered = (w * 0.40) < center_x < (w * 0.60)  # MUY centrado
        is_at_top = y < int(h * 0.08)  # MUY arriba (8% superior)
        
        # Debe tener UN ÁREA GRANDE y SER PROFUNDO
        significant_size = area > 50  # MÁS grande
        significant_depth = height > 12  # MÁS profundo
        
        # La hendidura debe ser más profunda que ancha (característica única)
        is_deep_notch = height > width * 1.2
        
        if is_centered and is_at_top and significant_size and significant_depth and is_deep_notch:
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

def detect_heart_pattern(img):
    """
    Detecta específicamente el patrón de corazón con características únicas
    """
    h, w = img.shape
    
    # Característica 1: Hendidura superior (MUY IMPORTANTE para corazones)
    has_notch, notch_depth = detect_top_notch(img)
    
    # Característica 2: Lóbulos redondeados
    has_rounded = detect_rounded_top(img)
    
    # Característica 3: Punta inferior centrada
    has_bottom, point_strength = detect_bottom_point(img)
    
    # Característica 4: Defectos de convexidad
    defects_count, max_depth, defect_depths = analyze_convexity_defects(img)
    has_some_defects = defects_count >= 1  # Más permisivo
    
    # Característica 5: Solidez media-baja (por la hendidura)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area = cv2.contourArea(cnt)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        medium_low_solidity = solidity < 0.93  # Más permisivo
    else:
        medium_low_solidity = False
        solidity = 0
    
    # Característica 6: Verificar 4 vértices
    epsilon1 = 0.01 * cv2.arcLength(cnt, True) if contours else 0
    epsilon2 = 0.02 * cv2.arcLength(cnt, True) if contours else 0
    epsilon3 = 0.03 * cv2.arcLength(cnt, True) if contours else 0
    
    approx1 = cv2.approxPolyDP(cnt, epsilon1, True) if contours else []
    approx2 = cv2.approxPolyDP(cnt, epsilon2, True) if contours else []
    approx3 = cv2.approxPolyDP(cnt, epsilon3, True) if contours else []
    
    has_4_vertices = (len(approx1) == 4) or (len(approx2) == 4) or (len(approx3) == 4)
    
    # Debug info
    print(f"    [HEART DEBUG] Hendidura: {has_notch} (depth: {notch_depth})")
    print(f"    [HEART DEBUG] Lóbulos redondeados: {has_rounded}")
    print(f"    [HEART DEBUG] Punta inferior: {has_bottom} (strength: {point_strength})")
    print(f"    [HEART DEBUG] Defectos: {defects_count} (max depth: {max_depth:.1f})")
    print(f"    [HEART DEBUG] Solidez: {solidity:.3f} (medium-low: {medium_low_solidity})")
    print(f"    [HEART DEBUG] Vértices: eps1={len(approx1)}, eps2={len(approx2)}, eps3={len(approx3)} (4: {has_4_vertices})")
    
    # Score ajustado - DOS CAMINOS
    score = 0.0
    
    # CAMINO 1: Si tiene hendidura (característica única del corazón)
    if has_notch and notch_depth > 15:
        score += 0.50  # MÁXIMA PRIORIDAD a hendidura profunda
        
        if medium_low_solidity:
            score += 0.20
        
        if not has_4_vertices:
            score += 0.15  # Bonus si NO es rombo perfecto
        
        if has_some_defects:
            score += 0.10
        
        if has_rounded:
            score += 0.05
    
    # CAMINO 2: Si NO tiene 4 vértices + otras características
    elif not has_4_vertices and medium_low_solidity:
        score += 0.35
        
        if has_some_defects:
            score += 0.20
        
        if has_notch:
            score += 0.15
        
        if has_rounded:
            score += 0.10
    
    # Es corazón SI tiene hendidura profunda O (no 4 vértices + solidez media-baja)
    is_heart = (has_notch and notch_depth > 15) or (not has_4_vertices and medium_low_solidity and has_some_defects)
    
    return is_heart, score

def detect_diamond_pattern(img):
    """
    Detecta específicamente el patrón de diamante
    """
    h, w = img.shape
    
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Aproximar polígono con diferentes epsilons
    epsilon1 = 0.01 * cv2.arcLength(cnt, True)
    epsilon2 = 0.02 * cv2.arcLength(cnt, True)
    epsilon3 = 0.03 * cv2.arcLength(cnt, True)
    
    approx1 = cv2.approxPolyDP(cnt, epsilon1, True)
    approx2 = cv2.approxPolyDP(cnt, epsilon2, True)
    approx3 = cv2.approxPolyDP(cnt, epsilon3, True)
    
    # CARACTERÍSTICA CLAVE: Debe tener 4 vértices (probar con diferentes aproximaciones)
    has_4_vertices = (len(approx1) == 4) or (len(approx2) == 4) or (len(approx3) == 4)
    
    # Calcular solidez (diamante DEBE tener MUY alta solidez)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(cnt)
    solidity = float(area) / hull_area if hull_area != 0 else 0
    very_high_solidity = solidity > 0.90  # MUY alta solidez
    
    # Debe tener MUY pocos defectos
    defects_count, max_depth, _ = analyze_convexity_defects(img)
    very_few_defects = defects_count <= 1  # Casi ninguno
    
    # NO debe tener lóbulos redondeados (eso es corazón)
    has_rounded = detect_rounded_top(img)
    no_rounded = not has_rounded
    
    # Verificar hendidura (MENOS PRIORITARIO que 4 vértices + solidez)
    has_notch, notch_depth = detect_top_notch(img)
    
    # Debug
    print(f"    [DIAMOND DEBUG] Vértices: eps1={len(approx1)}, eps2={len(approx2)}, eps3={len(approx3)} (4: {has_4_vertices})")
    print(f"    [DIAMOND DEBUG] Solidez: {solidity:.3f} (very high: {very_high_solidity})")
    print(f"    [DIAMOND DEBUG] Defectos: {defects_count} (very few: {very_few_defects})")
    print(f"    [DIAMOND DEBUG] Hendidura: {has_notch} (depth: {notch_depth})")
    print(f"    [DIAMOND DEBUG] Sin lóbulos: {no_rounded}")
    
    # Score - PRIORIDAD ABSOLUTA a 4 vértices + alta solidez
    score = 0.0
    
    # Si tiene 4 vértices + alta solidez + pocos defectos = ES DIAMANTE
    if has_4_vertices and very_high_solidity and very_few_defects:
        score += 0.70  # MÁXIMA PRIORIDAD
        
        if no_rounded:
            score += 0.15
        
        if not has_notch:  # Bonus si NO tiene hendidura
            score += 0.10
        
        if solidity > 0.93:  # Bonus por solidez muy alta
            score += 0.05
    
    # Diamante SI Y SOLO SI tiene 4 vértices + muy alta solidez + muy pocos defectos
    is_diamond = (has_4_vertices and very_high_solidity and very_few_defects)
    
    return is_diamond, score

def detect_spade_pattern(img):
    """
    Detecta específicamente el patrón de pica
    """
    h, w = img.shape
    
    # NO debe tener hendidura superior (eso es corazón)
    has_notch, _ = detect_top_notch(img)
    no_notch = not has_notch
    
    # Analizar defectos (pica tiene moderados, menos que trébol)
    defects_count, max_depth, _ = analyze_convexity_defects(img)
    moderate_defects = 2 <= defects_count <= 4
    
    # Solidez media-alta (más que trébol, menos que diamante)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area = cv2.contourArea(cnt)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        medium_high_solidity = 0.82 < solidity < 0.92
    else:
        medium_high_solidity = False
        solidity = 0
    
    # CARACTERÍSTICA CLAVE: Detectar PUNTA ÚNICA en el top
    # Pica tiene una sola punta aguda arriba
    # Trébol tiene superficie plana/ondulada (3 círculos)
    top_10 = img[:int(h*0.1), :]
    
    # Proyección horizontal del top
    top_projection = np.sum(top_10, axis=0)
    
    if len(top_projection) > 0 and np.max(top_projection) > 0:
        # Normalizar
        top_projection = top_projection / np.max(top_projection)
        
        # Buscar un pico único y pronunciado (punta de pica)
        # vs múltiples picos o superficie plana (trébol)
        
        # Contar cuántas columnas tienen más del 50% de intensidad
        high_intensity_cols = np.sum(top_projection > 0.5)
        total_cols = len(top_projection)
        
        # Pica: pocas columnas con alta intensidad (punta aguda)
        # Trébol: muchas columnas con alta intensidad (3 círculos = ancho)
        narrow_top = (high_intensity_cols / total_cols) < 0.3
        
        # Buscar el ancho del pico
        # Pica: pico estrecho en el centro
        # Trébol: ancho casi completo
        center = len(top_projection) // 2
        center_range = range(max(0, center-10), min(len(top_projection), center+10))
        center_mass = np.sum([top_projection[i] for i in center_range if i < len(top_projection)])
        total_mass = np.sum(top_projection)
        
        # Pica: masa concentrada en el centro
        centered_peak = (center_mass / total_mass) > 0.4 if total_mass > 0 else False
    else:
        narrow_top = False
        centered_peak = False
    
    # Analizar la forma del top más detalladamente
    top_25 = img[:int(h*0.25), :]
    top_contours, _ = cv2.findContours(top_25, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if top_contours:
        top_cnt = max(top_contours, key=cv2.contourArea)
        
        # Encontrar el punto más alto
        topmost = tuple(top_cnt[top_cnt[:, :, 1].argmin()][0])
        
        # Verificar que está centrado (característica de pica)
        is_top_centered = abs(topmost[0] - w//2) < w * 0.2
    else:
        is_top_centered = False
    
    # Debug
    print(f"    [SPADE DEBUG] Defectos: {defects_count} (max depth: {max_depth:.1f})")
    print(f"    [SPADE DEBUG] Solidez: {solidity:.3f} (medium-high: {medium_high_solidity})")
    print(f"    [SPADE DEBUG] Top estrecho: {narrow_top}")
    print(f"    [SPADE DEBUG] Pico centrado: {centered_peak}")
    print(f"    [SPADE DEBUG] Punto superior centrado: {is_top_centered}")
    
    # Score ajustado
    score = 0.0
    
    if narrow_top and centered_peak:
        score += 0.45  # CRÍTICO: punta única y estrecha
    
    if medium_high_solidity:
        score += 0.25
    
    if moderate_defects:
        score += 0.15
    
    if is_top_centered:
        score += 0.10
    
    if no_notch:
        score += 0.05
    
    is_spade = (narrow_top and centered_peak and medium_high_solidity)
    
    return is_spade, score

def detect_club_pattern(img):
    """
    Detecta específicamente el patrón de trébol
    """
    h, w = img.shape
    
    # Analizar defectos de convexidad (trébol tiene muchos)
    defects_count, max_depth, defect_depths = analyze_convexity_defects(img)
    many_defects = defects_count >= 5
    
    # Muy baja solidez (por las separaciones entre círculos)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        area = cv2.contourArea(cnt)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        very_low_solidity = solidity < 0.82
    else:
        very_low_solidity = False
        solidity = 0
    
    # CARACTERÍSTICA CLAVE: TOP ANCHO Y PLANO (3 círculos)
    top_10 = img[:int(h*0.1), :]
    
    # Proyección horizontal del top
    top_projection = np.sum(top_10, axis=0)
    
    wide_top = False
    has_undulations = False
    high_intensity_cols = 0
    total_cols = 1  # Inicializar para evitar división por cero
    
    if len(top_projection) > 0 and np.max(top_projection) > 0:
        # Normalizar
        top_projection = top_projection / np.max(top_projection)
        
        # Contar cuántas columnas tienen alta intensidad
        high_intensity_cols = np.sum(top_projection > 0.5)
        total_cols = len(top_projection)
        
        # Trébol: MUCHAS columnas con intensidad (ancho por los 3 círculos)
        wide_top = (high_intensity_cols / total_cols) > 0.4
        
        # Buscar múltiples ondulaciones (picos y valles por los círculos)
        # Calcular segunda derivada (cambios de pendiente)
        if len(top_projection) > 2:
            diff = np.diff(top_projection)
            
            # Contar cambios de signo (de subir a bajar = pico)
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
            
            # Trébol: múltiples cambios (ondulaciones por círculos)
            has_undulations = sign_changes >= 4
    
    # Distribución: muy top-heavy
    top_70 = img[:int(h*0.7), :]
    bottom_30 = img[int(h*0.7):, :]
    top_pixels = cv2.countNonZero(top_70)
    bottom_pixels = cv2.countNonZero(bottom_30)
    total = top_pixels + bottom_pixels
    very_top_heavy = (top_pixels / total > 0.70) if total > 0 else False
    
    # Analizar ancho en la parte superior vs medio
    top_30 = img[:int(h*0.3), :]
    middle_30 = img[int(h*0.3):int(h*0.6), :]
    
    # Ancho promedio (columnas no vacías)
    top_cols = [i for i in range(w) if np.sum(top_30[:, i]) > 0]
    middle_cols = [i for i in range(w) if np.sum(middle_30[:, i]) > 0]
    
    top_width = len(top_cols)
    middle_width = len(middle_cols)
    
    # Trébol: top es IGUAL O MÁS ANCHO que middle (por los círculos)
    # Pica: top es MÁS ESTRECHO que middle (se va estrechando hacia la punta)
    top_wider_or_equal = top_width >= middle_width * 0.9 if middle_width > 0 else False
    
    # Debug
    print(f"    [CLUB DEBUG] Defectos: {defects_count} (many: {many_defects})")
    print(f"    [CLUB DEBUG] Solidez: {solidity:.3f} (very low: {very_low_solidity})")
    print(f"    [CLUB DEBUG] Top ancho: {wide_top} (ratio: {high_intensity_cols/total_cols:.2f})")
    print(f"    [CLUB DEBUG] Ondulaciones: {has_undulations}")
    print(f"    [CLUB DEBUG] Top más ancho que medio: {top_wider_or_equal} (top:{top_width} vs mid:{middle_width})")
    print(f"    [CLUB DEBUG] Very top heavy: {very_top_heavy}")
    
    # Score ajustado
    score = 0.0
    
    if wide_top and top_wider_or_equal:
        score += 0.40  # CRÍTICO: top ancho (3 círculos)
    
    if very_low_solidity:
        score += 0.25
    
    if has_undulations:
        score += 0.15  # Ondulaciones por los círculos
    
    if many_defects:
        score += 0.10
    
    if very_top_heavy:
        score += 0.10
    
    is_club = (wide_top and very_low_solidity and top_wider_or_equal)
    
    return is_club, score

def classify_suit_by_pattern(img, color):
    """
    Clasifica el palo usando detección de patrones específicos
    """
    if color == 'red':
        # Detectar diamante y corazón
        is_diamond, diamond_score = detect_diamond_pattern(img)
        is_heart, heart_score = detect_heart_pattern(img)
        
        print(f"  [PATTERN] Diamante: {is_diamond} (score: {diamond_score:.3f})")
        print(f"  [PATTERN] Corazón: {is_heart} (score: {heart_score:.3f})")
        
        # IMPORTANTE: Devolver basándose en el score más alto
        if diamond_score > heart_score:
            return 'diamonds', diamond_score
        else:
            return 'hearts', heart_score
            
    else:  # black
        # Detectar pica y trébol
        is_spade, spade_score = detect_spade_pattern(img)
        is_club, club_score = detect_club_pattern(img)
        
        print(f"  [PATTERN] Pica: {is_spade} (score: {spade_score:.3f})")
        print(f"  [PATTERN] Trébol: {is_club} (score: {club_score:.3f})")
        
        if club_score > spade_score:
            return 'clubs', club_score
        else:
            return 'spades', spade_score

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
    """
    # PASO 1: Detección por patrones geométricos (90% peso)
    suit_by_pattern, pattern_score = classify_suit_by_pattern(img, color)
    
    print(f"  Clasificación por patrón: {suit_by_pattern} (score: {pattern_score:.3f})")
    
    # PASO 2: Template matching (10% peso, solo confirmación)
    templates = SUIT_TEMPLATES.get(color, {})
    template_scores = {}
    
    for name, tmpl in templates.items():
        score = match_template_multi_method(img, tmpl)
        template_scores[name] = score
    
    # Combinar scores (90% patrón, 10% template)
    final_scores = {}
    for name in template_scores.keys():
        pattern_score_for_name = pattern_score if name == suit_by_pattern or name.startswith(suit_by_pattern) else 0.0
        temp_score = template_scores[name]
        final_scores[name] = 0.90 * pattern_score_for_name + 0.10 * temp_score
    
    if final_scores:
        best_suit = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_suit]
        return best_suit, best_score, final_scores
    
    return suit_by_pattern, pattern_score, {suit_by_pattern: pattern_score}

def recognize_suit(warp):
    """
    Reconoce el palo de una carta desde su imagen warpeada
    """
    suit_roi = warp[100:170, 10:60]
    
    # PASO 1: Determinar color
    is_red = is_red_suit(suit_roi)
    color = 'red' if is_red else 'black'
    
    print(f"\n--- Detectando palo ---")
    print(f"Color detectado: {'ROJO' if is_red else 'NEGRO'}")
    
    # Convertir a escala de grises y binarizar
    suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    _, suit_bin = cv2.threshold(suit_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Limpiar ruido
    kernel = np.ones((3,3), np.uint8)
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_OPEN, kernel, iterations=2)
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Extraer contorno principal
    contours, _ = cv2.findContours(suit_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        mask = np.zeros_like(suit_bin)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        suit_bin = cv2.bitwise_and(suit_bin, mask)
    
    # Redimensionar
    suit_bin = cv2.resize(suit_bin, (70, 70))
    
    # PASO 2: Reconocimiento con múltiples rotaciones
    best_suit_name = None
    best_suit_score = -1
    best_angle = 0
    all_results = {}
    
    for angle in [0, 90, 180, 270]:
        print(f"\n=== Rotación {angle}° ===")
        if angle > 0:
            M = cv2.getRotationMatrix2D((35, 35), angle, 1.0)
            suit_rotated = cv2.warpAffine(suit_bin, M, (70, 70))
        else:
            suit_rotated = suit_bin
        
        suit_name, suit_score, scores_dict = match_suit_by_color(suit_rotated, color)
        
        all_results[angle] = scores_dict
        print(f"  Resultado: {suit_name} (score: {suit_score:.3f})")
        
        if suit_score > best_suit_score:
            best_suit_score = suit_score
            best_suit_name = suit_name
            best_angle = angle
    
    print(f"\n*** MEJOR MATCH: {best_suit_name} en {best_angle}° (score: {best_suit_score:.3f}) ***")
    
    return best_suit_name, best_suit_score