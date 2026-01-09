import cv2
import numpy as np

def smart_resize_for_ocr(image_path):
    """
    Redimensiona automáticamente para tamaño óptimo de OCR.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    print(f"📐 Imagen original: {w}x{h}")
    
    # Si es muy pequeña, aumentar
    if w < 200:
        target_width = 300
        scale_factor = target_width / w
        new_w = target_width
        new_h = int(h * scale_factor)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"✅ Aumentada a: {new_w}x{new_h}")
        return resized
    
    # Si es muy grande, reducir
    elif w > 800:
        target_width = 600
        scale_factor = target_width / w
        new_w = target_width
        new_h = int(h * scale_factor)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"✅ Reducida a: {new_w}x{new_h}")
        return resized
    
    # Tamaño óptimo
    else:
        print("✅ Tamaño óptimo")
        return img