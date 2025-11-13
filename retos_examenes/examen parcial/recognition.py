import cv2
import numpy as np
import os

def load_templates(path, size=(70,100)):

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
ranks = load_templates('templates/ranks', size=(70,100))
suits = load_templates('templates/suits', size=(30,30))

def match_symbol(img, templates):

    best_score = -1
    best_name = None
    for name, tmpl in templates.items():
        res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score

def recognize_card(warp):

    h, w = warp.shape[:2]

    # --- Región del rank (número/letra) ---
    rank_roi = warp[10:110, 10:80]  # ajustar según tamaño de tu warp
    rank_gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
    _, rank_bin = cv2.threshold(rank_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rank_bin = cv2.resize(rank_bin, (70,100))

    # --- Región del suit (símbolo) ---
    suit_roi = warp[80:120, 10:40]  # debajo del número
    suit_gray = cv2.cvtColor(suit_roi, cv2.COLOR_BGR2GRAY)
    _, suit_bin = cv2.threshold(suit_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    suit_bin = cv2.resize(suit_bin, (30,30))

    # Matching
    rank_name, rank_score = match_symbol(rank_bin, ranks)
    suit_name, suit_score = match_symbol(suit_bin, suits)

    return rank_name, suit_name, rank_score, suit_score
