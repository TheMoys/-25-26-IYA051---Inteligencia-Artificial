import cv2
import numpy as np
import os

def load_templates(path):
    templates = {}
    for name in os.listdir(path):
        if not name.endswith('.png'): 
            continue
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (70, 100))
        templates[os.path.splitext(name)[0]] = img
    return templates

ranks = load_templates('templates/ranks')
suits = load_templates('templates/suits')

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
    # Recortar esquina superior izquierda
    corner = warp[10:130, 10:80]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    rank_name, rank_score = match_symbol(thresh, ranks)
    suit_name, suit_score = match_symbol(thresh, suits)

    return rank_name, suit_name, rank_score, suit_score
