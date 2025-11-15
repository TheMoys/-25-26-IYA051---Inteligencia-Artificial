import cv2
import numpy as np
import os

def load_suit_templates(path='templates/suits', size=(40,40)):
    """Carga las plantillas de los palos"""
    templates = {}
    for name in os.listdir(path):
        if not name.lower().endswith('.png'):
            continue
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        templates[os.path.splitext(name)[0]] = img_bin
    return templates

# Cargar plantillas al inicio
SUIT_TEMPLATES = load_suit_templates()

def is_red_suit(suit_roi):
    """
    Determina si el palo es rojo (corazones/diamantes) o negro (picas/tréboles)
    analizando solo los píxeles del símbolo, no el fondo blanco
    """
    # Convertir a escala de grises para segmentar el símbolo
    gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    
    # Crear máscara del símbolo (píxeles oscuros = símbolo)
    _, symbol_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Limpiar la máscara
    kernel = np.ones((3,3), np.uint8)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_OPEN, kernel)
    symbol_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel)
    
    # Verificar que hay suficientes píxeles de símbolo
    if cv2.countNonZero(symbol_mask) < 50:
        return False  # Por defecto negro si no hay símbolo claro
    
    # Extraer solo los píxeles del símbolo
    hsv = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(suit_roi)
    
    # Aplicar la máscara para analizar solo el símbolo
    r_symbol = cv2.bitwise_and(r, r, mask=symbol_mask)
    b_symbol = cv2.bitwise_and(b, b, mask=symbol_mask)
    g_symbol = cv2.bitwise_and(g, g, mask=symbol_mask)
    
    # Método 1: Análisis HSV solo en píxeles del símbolo
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
    
    # Método 2: Análisis RGB en píxeles del símbolo
    mask_rgb = np.zeros(r.shape, dtype=np.uint8)
    mask_rgb[(r > b + 20) & (r > g) & (r > 80)] = 255
    mask_rgb = cv2.bitwise_and(mask_rgb, symbol_mask)
    red_pixels_rgb = cv2.countNonZero(mask_rgb)
    ratio_rgb = red_pixels_rgb / symbol_pixels if symbol_pixels > 0 else 0
    
    # Método 3: Comparación de medias solo en el símbolo
    mean_r = np.sum(r_symbol) / symbol_pixels if symbol_pixels > 0 else 0
    mean_b = np.sum(b_symbol) / symbol_pixels if symbol_pixels > 0 else 0
    mean_g = np.sum(g_symbol) / symbol_pixels if symbol_pixels > 0 else 0
    
    # Debug
    print(f"HSV ratio: {ratio_hsv:.3f}, RGB ratio: {ratio_rgb:.3f}, R: {mean_r:.1f}, G: {mean_g:.1f}, B: {mean_b:.1f}")
    
    # Criterios ajustados
    is_red_hsv = ratio_hsv > 0.08
    is_red_rgb = ratio_rgb > 0.12
    is_red_channel = (mean_r - mean_b) > 15 and (mean_r - mean_g) > 10
    
    # Votación: 2 de 3
    votes = sum([is_red_hsv, is_red_rgb, is_red_channel])
    
    print(f"Votos: HSV={is_red_hsv}, RGB={is_red_rgb}, Channels={is_red_channel} -> {'RED' if votes >= 2 else 'BLACK'}")
    
    return votes >= 2

def match_suit_template(img, color_filter=None):
    """
    Encuentra la mejor coincidencia entre las plantillas de palos
    color_filter: 'red' para corazones/diamantes, 'black' para picas/tréboles, None para todos
    """
    best_score = -1
    best_name = None
    
    for name, tmpl in SUIT_TEMPLATES.items():
        # Filtrar por color si se especifica
        if color_filter == 'red' and name.lower() not in ['hearts', 'diamonds', 'corazones', 'diamantes', 'oros']:
            continue
        if color_filter == 'black' and name.lower() not in ['spades', 'clubs', 'picas', 'treboles', 'espadas', 'bastos']:
            continue
            
        res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_score:
            best_score = score
            best_name = name
    
    return best_name, best_score

def recognize_suit(warp):
    """
    Reconoce el palo de una carta desde su imagen warpeada
    Retorna: (suit_name, suit_score)
    """
    # Expandir la región del suit para capturar mejor el color
    suit_roi = warp[100:170, 10:60]
    
    # Determinar si es rojo o negro
    is_red = is_red_suit(suit_roi)
    color_filter = 'red' if is_red else 'black'
    
    # Debug: guardar ROI para verificar
    cv2.imwrite(f'debug_suit_roi_{color_filter}.png', suit_roi)
    
    # Convertir a escala de grises y binarizar
    suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    
    # Usar umbralización adaptativa para mejor binarización
    suit_bin = cv2.adaptiveThreshold(suit_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    # Limpiar ruido
    kernel = np.ones((3,3), np.uint8)
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_CLOSE, kernel)
    suit_bin = cv2.morphologyEx(suit_bin, cv2.MORPH_OPEN, kernel)
    
    suit_bin = cv2.resize(suit_bin, (40, 40))
    
    # Aplicar múltiples rotaciones para mejor matching
    best_suit_name = None
    best_suit_score = -1
    
    for angle in [0, 90, 180, 270]:
        if angle > 0:
            M = cv2.getRotationMatrix2D((20, 20), angle, 1.0)
            suit_rotated = cv2.warpAffine(suit_bin, M, (40, 40))
        else:
            suit_rotated = suit_bin
            
        suit_name, suit_score = match_suit_template(suit_rotated, color_filter)
        if suit_score > best_suit_score:
            best_suit_score = suit_score
            best_suit_name = suit_name
    
    return best_suit_name, best_suit_score