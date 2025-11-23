import cv2
import numpy as np
import sys
import io

# Parámetros (ajustables)
MIN_CARD_AREA = 2000        
WARP_WIDTH = 300            
WARP_HEIGHT = 450           

# Parámetros de visualización
DISPLAY_WIDTH = 1280        
DISPLAY_HEIGHT = 720        

# NUEVO: Variable global para controlar debug
_DEBUG_ENABLED = False
_original_stdout = sys.stdout
_original_imwrite = None


def enable_debug():
    """Habilita prints y guardado de imágenes"""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = True

def disable_debug():
    """Deshabilita prints y guardado de imágenes"""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = False

def _suppress_output():
    """Suprime prints y cv2.imwrite temporalmente"""
    global _original_imwrite
    
    # Suprimir stdout
    sys.stdout = io.StringIO()
    
    # Monkey-patch cv2.imwrite para que no haga nada
    if _original_imwrite is None:
        _original_imwrite = cv2.imwrite
    cv2.imwrite = lambda *args, **kwargs: True

def _restore_output():
    """Restaura prints y cv2.imwrite"""
    global _original_imwrite
    
    # Restaurar stdout
    sys.stdout = _original_stdout
    
    # Restaurar cv2.imwrite
    if _original_imwrite is not None:
        cv2.imwrite = _original_imwrite

def open_video_source(source):
    if isinstance(source, str) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        cap = cv2.VideoCapture(source)
        # IMAGEN ESTÁTICA: Habilitar debug
        enable_debug()
        return cap, True

    if source == 'webcam':
        cap = cv2.VideoCapture(0)
        # VIDEO: Deshabilitar debug
        disable_debug()
        return cap, False

    cap = cv2.VideoCapture(source)
    # VIDEO: Deshabilitar debug
    disable_debug()
    return cap, False

def resize_to_display(frame, max_width=DISPLAY_WIDTH, max_height=DISPLAY_HEIGHT):
    """
    Redimensiona el frame para que quepa en el display manteniendo el aspect ratio
    """
    h, w = frame.shape[:2]
    
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)
    
    if scale >= 1.0:
        return frame
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized

def order_points(pts):
    """
    Ordena los 4 puntos en orden: top-left, top-right, bottom-right, bottom-left
    """
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    dists = np.sum(sorted_pts ** 2, axis=1)
    tl_idx = np.argmin(dists)
    sorted_pts = np.roll(sorted_pts, -tl_idx, axis=0)
    
    return sorted_pts.astype("float32")

def four_point_transform(image, pts, w=WARP_WIDTH, h=WARP_HEIGHT):
    """
    Transforma perspectiva y GARANTIZA que la carta quede VERTICAL
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width = max(int(width_top), int(width_bottom))
    
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height = max(int(height_left), int(height_right))
    
    if width > height:
        dst = np.array([
            [0, h - 1],
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1]
        ], dtype="float32")
    else:
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (w, h))
    
    return warp

def extract_card_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cards = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CARD_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2).astype("float32")
            cards.append(pts)
        else:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype("float32")
            cards.append(box)
    return cards

def segment_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_clean

def process_frame(frame):
    """
    Procesa el frame
    - Si debug está DESHABILITADO (video): Suprime prints y archivos
    - Si debug está HABILITADO (imagen): Funciona normal
    """
    frame = resize_to_display(frame)
    
    out = frame.copy()
    h, w = frame.shape[:2]

    mask = segment_green(frame)
    card_polys = extract_card_contours(mask)

    warps = []

    for pts in card_polys:
        try:
            # SUPRIMIR OUTPUT SI DEBUG ESTÁ DESHABILITADO
            if not _DEBUG_ENABLED:
                _suppress_output()
            
            from recognition_rank import recognize_rank
            from recognition_suits import recognize_suit

            warp = four_point_transform(frame, pts, w=WARP_WIDTH, h=WARP_HEIGHT)
            warps.append(warp)

            pts_int = pts.reshape((-1,1,2)).astype(int)
            M = cv2.moments(pts_int)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                cx = int(np.mean(pts[:,0]))
                cy = int(np.mean(pts[:,1]))

            cv2.polylines(out, [pts_int], True, (0, 255, 255), 2)
            cv2.circle(out, (cx, cy), 4, (0,0,255), -1)

            rank, rs = recognize_rank(warp)
            suit, ss = recognize_suit(warp)
            
            # RESTAURAR OUTPUT SI ESTABA SUPRIMIDO
            if not _DEBUG_ENABLED:
                _restore_output()
                # Imprimir solo el resultado final
                print(f"✓ Carta: {rank} de {suit}")
            
            cv2.putText(out, f'{rank} de {suit}', (cx-50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        except Exception as e:
            # RESTAURAR OUTPUT en caso de error
            if not _DEBUG_ENABLED:
                _restore_output()
            print(f"Error: {e}")
            continue

    if len(warps) > 0:
        thumb_h = WARP_HEIGHT
        thumb_w = WARP_WIDTH
        max_thumbs = max(1, h // thumb_h)
        thumbs = np.zeros((h, thumb_w, 3), dtype=np.uint8) + 50
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

    info = f'Cartas detectadas: {len(warps)}'
    cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    combined = resize_to_display(combined)

    return combined