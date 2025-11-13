import cv2
import numpy as np

# Parámetros (ajustables)
MIN_CARD_AREA = 2000        # area mínima del contorno para considerarlo carta (ajusta si tus cartas salen pequeñas)
WARP_WIDTH = 300            # ancho del ROI final de la carta
WARP_HEIGHT = 450           # alto del ROI final de la carta


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

def order_points(pts):
    # pts: array de 4 puntos (x,y)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]    # top-left
    rect[2] = pts[np.argmax(s)]    # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def four_point_transform(image, pts, w=WARP_WIDTH, h=WARP_HEIGHT):
    rect = order_points(pts)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (w,h))
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

    # procesa un frame y devuelve el frame con anotaciones
    out = frame.copy()
    h, w = frame.shape[:2]

    # Segmentar fondo verde (obtener máscara de objetos)
    mask = segment_green(frame)

    # Extraer contornos de posibles cartas
    card_polys = extract_card_contours(mask)

    warps = []
    for pts in card_polys:
        try:
            warp = four_point_transform(frame, pts, w=WARP_WIDTH, h=WARP_HEIGHT)
            warps.append(warp)
        except Exception as e:
            # fallback: ignora esta carta si la transformada falla
            print("Warpping failed:", e)
            continue

    # Dibujar contornos detectados en el frame original
    for pts in card_polys:
        pts_int = pts.reshape((-1,1,2)).astype(int)
        cv2.polylines(out, [pts_int], True, (0, 255, 255), 2)
        # dibujar el centro
        M = cv2.moments(pts_int)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            cv2.circle(out, (cx, cy), 4, (0,0,255), -1)

    # Crear una columna a la derecha para miniaturas (si hay warps)
    if len(warps) > 0:
        thumb_h = WARP_HEIGHT
        thumb_w = WARP_WIDTH
        # limitar número de miniaturas a lo que quepa en la ventana vertical
        max_thumbs = max(1, h // thumb_h)
        # componer una imagen vertical con miniaturas (hasta max_thumbs)
        thumbs = np.zeros((h, thumb_w, 3), dtype=np.uint8) + 50  # fondo gris
        for i, wp in enumerate(warps[:max_thumbs]):
            y0 = i * thumb_h
            y1 = y0 + thumb_h
            if y1 > h:
                break
            # si el warp tiene distinto tamaño, redimensionar
            wp_small = cv2.resize(wp, (thumb_w, thumb_h))
            thumbs[y0:y1, 0:thumb_w] = wp_small

        # concatenar horizontalmente
        combined = np.hstack((out, thumbs))
    else:
        combined = out

    # Mostrar texto informativo
    info = f'Cartas detectadas: {len(warps)}'
    cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    return combined
