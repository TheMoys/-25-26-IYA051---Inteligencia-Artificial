import cv2
import numpy as np

# Parámetros (ajustables)
MIN_CARD_AREA = 2000        # area mínima del contorno para considerarlo carta (ajusta si tus cartas salen pequeñas)
WARP_WIDTH = 300            # ancho del ROI final de la carta
WARP_HEIGHT = 450           # alto del ROI final de la carta

# Parámetros de visualización
DISPLAY_WIDTH = 1280        # Ancho estándar del display
DISPLAY_HEIGHT = 720        # Alto estándar del display


def open_video_source(source):
    # Devuelve objeto VideoCapture y flag is_image
    if isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        cap = cv2.VideoCapture(source)
        return cap, True

    if source == 'webcam':
        cap = cv2.VideoCapture(0)
        return cap, False

    cap = cv2.VideoCapture(source)
    return cap, False

def resize_to_display(frame, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT):
    """
    Redimensiona el frame para que quepa en el display manteniendo el aspect ratio
    """
    h, w = frame.shape[:2]
    
    # Calcular el factor de escala para que quepa en el display
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    # Si la imagen ya es más pequeña que el display, no redimensionar
    if scale >= 1.0:
        return frame
    
    # Calcular nuevas dimensiones
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionar
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

def order_points(pts):
    """
    Ordena los 4 puntos en orden: top-left, top-right, bottom-right, bottom-left
    MEJORADO: Usa centroide y ángulos para manejar rotaciones extremas
    """
    # Calcular centroide
    center = pts.mean(axis=0)
    
    # Calcular ángulos desde el centro (en radianes)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Ordenar puntos por ángulo (sentido antihorario desde la derecha)
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # Encontrar el punto más cercano a la esquina superior izquierda (0, 0)
    dists = np.sum(sorted_pts ** 2, axis=1)
    tl_idx = np.argmin(dists)
    
    # Rotar array para que top-left sea el primero
    sorted_pts = np.roll(sorted_pts, -tl_idx, axis=0)
    
    return sorted_pts.astype("float32")

def four_point_transform(image, pts, w=WARP_WIDTH, h=WARP_HEIGHT):
    """
    Transforma perspectiva y GARANTIZA que la carta quede VERTICAL
    """
    # Ordenar puntos
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Calcular dimensiones del cuadrilátero
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width = max(int(width_top), int(width_bottom))
    
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height = max(int(height_left), int(height_right))
    
    # Determinar si está horizontal o vertical
    if width > height:
        # Está HORIZONTAL → Necesita rotación 90°
        # Intercambiar dimensiones y rotar puntos
        dst = np.array([
            [0, h - 1],           # tl → bl (rotado 90° antihorario)
            [0, 0],               # tr → tl
            [w - 1, 0],           # br → tr
            [w - 1, h - 1]        # bl → br
        ], dtype="float32")
    else:
        # Está VERTICAL → Usar orden normal
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")
    
    # Aplicar transformación de perspectiva
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (w, h))
    
    return warp

def extract_card_contours(mask):
    # encuentra contornos y devuelve lista de contornos útiles
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CARD_AREA:
            continue
        # aproximar contorno
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype("float32")
            cards.append(pts)
        else:
            # si no tiene 4 vértices, tomar el rectángulo mínimo y usar sus 4 puntos
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype("float32")
            cards.append(box)
    return cards

def segment_green(frame):
    # Convierte a HSV y umbraliza color verde del tapete
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Rangos típicos de verde
    lower = np.array([35, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Invertir la máscara para obtener objetos que no son verde (las cartas)
    mask_inv = cv2.bitwise_not(mask)
    # limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_clean

def process_frame(frame):
    # Redimensionar frame si es muy grande
    frame = resize_to_display(frame)
    
    out = frame.copy()
    h, w = frame.shape[:2]

    # Segmentar fondo verde
    mask = segment_green(frame)

    # Extraer contornos de posibles cartas
    card_polys = extract_card_contours(mask)

    warps = []

    for pts in card_polys:
        try:
            from recognition_rank import recognize_rank
            from recognition_suits import recognize_suit

            # Warp de la carta
            warp = four_point_transform(frame, pts, w=WARP_WIDTH, h=WARP_HEIGHT)
            warps.append(warp)

            # Calcular centro del contorno para colocar texto
            pts_int = pts.reshape((-1,1,2)).astype(int)
            M = cv2.moments(pts_int)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                cx = int(np.mean(pts[:,0]))
                cy = int(np.mean(pts[:,1]))

            # Dibujar contorno y centro
            cv2.polylines(out, [pts_int], True, (0, 255, 255), 2)
            cv2.circle(out, (cx, cy), 4, (0,0,255), -1)

            # Reconocimiento de la carta
            rank, rs = recognize_rank(warp)
            suit, ss = recognize_suit(warp)
            
            cv2.putText(out, f'{rank} de {suit}', (cx-50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        except Exception as e:
            print("Warpping failed:", e)
            continue

    # Crear columna de miniaturas a la derecha
    if len(warps) > 0:
        thumb_h = WARP_HEIGHT
        thumb_w = WARP_WIDTH
        max_thumbs = max(1, h // thumb_h)
        thumbs = np.zeros((h, thumb_w, 3), dtype=np.uint8) + 50  # fondo gris
        for i, wp in enumerate(warps[:max_thumbs]):
            y0 = i * thumb_h
            y1 = y0 + thumb_h
            if y1 > h:
                break
            wp_small = cv2.resize(wp, (thumb_w, thumb_h))
            thumbs[y0:y1, 0:thumb_w] = wp_small
        combined = np.hstack((out, thumbs))
    else:
        combined = out

    # Texto informativo
    info = f'Cartas detectadas: {len(warps)}'
    cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # Asegurar que el resultado final también quepa en el display
    combined = resize_to_display(combined)

    return combined