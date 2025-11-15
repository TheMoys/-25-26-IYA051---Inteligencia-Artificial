import cv2
import numpy as np
import os

def load_rank_templates(path='templates/ranks', size=(70,100)):
    """Carga las plantillas de los números/letras"""
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
RANK_TEMPLATES = load_rank_templates()

def match_rank_template(img):
    """
    Encuentra la mejor coincidencia entre las plantillas de rangos
    """
    best_score = -1
    best_name = None
    
    for name, tmpl in RANK_TEMPLATES.items():
        res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_score:
            best_score = score
            best_name = name
    
    return best_name, best_score

def recognize_rank(warp):
    """
    Reconoce el número/letra de una carta desde su imagen warpeada
    Retorna: (rank_name, rank_score)
    """
    # --- Región del rank (número/letra) ---
    rank_roi = warp[10:110, 10:80]
    rank_gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)
    _, rank_bin = cv2.threshold(rank_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rank_bin = cv2.resize(rank_bin, (70, 100))
    
    # Matching del rank
    rank_name, rank_score = match_rank_template(rank_bin)
    
    return rank_name, rank_score