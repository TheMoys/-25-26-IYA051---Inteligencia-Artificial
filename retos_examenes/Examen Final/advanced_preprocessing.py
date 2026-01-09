import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def advanced_char_preprocessing(img_array, target_size=(32, 32), debug=False):
    """
    Preprocesamiento EXTREMADAMENTE robusto para caracteres.
    """
    # Asegurar escala de grises
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    original_shape = img_array.shape
    
    # 1. Corrección de contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    img_enhanced = clahe.apply(img_array)
    
    # 2. Detección automática de inversión
    mean_val = np.mean(img_enhanced)
    if mean_val > 127:  # Fondo claro
        img_enhanced = 255 - img_enhanced
    
    # 3. Binarización Otsu + limpieza morfológica
    _, binary = cv2.threshold(img_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Eliminación de ruido pequeño
    kernel_noise = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_noise)
    
    # 5. Encontrar el bounding box del carácter real
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        # Si no hay contenido, crear imagen vacía
        result = np.zeros(target_size, dtype=np.float32)
        if debug:
            print("⚠️ Imagen vacía después del preprocesamiento")
        return result
    
    # Calcular bounding box apretado
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Extraer solo el carácter
    char_tight = binary[y_min:y_max+1, x_min:x_max+1]
    
    # 6. Normalización de aspecto con padding inteligente
    h, w = char_tight.shape
    
    # Calcular padding para hacer cuadrado
    max_dim = max(h, w)
    
    # Añadir 20% de padding
    padded_size = int(max_dim * 1.2)
    
    # Centrar el carácter
    pad_h = (padded_size - h) // 2
    pad_w = (padded_size - w) // 2
    
    padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = char_tight
    
    # 7. Redimensionar con interpolación suave
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    
    # 8. Normalización final
    resized_normalized = resized / 255.0
    
    # 9. Mejora final del contraste
    resized_normalized = np.clip(resized_normalized * 1.2, 0, 1)
    
    if debug:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(binary, cmap='gray')
        axes[1].set_title('Binarizado')
        axes[2].imshow(char_tight, cmap='gray')
        axes[2].set_title('Tight BB')
        axes[3].imshow(resized_normalized, cmap='gray')
        axes[3].set_title('Final 32x32')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return resized_normalized.astype(np.float32)

def intelligent_segmentation(image_path, debug=False):
    """
    Segmentación mejorada para manuscritas.
    """
    # NUEVO: Usar redimensionamiento automático
    img = smart_resize_for_ocr(image_path)
    h, w = img.shape
    
    # Detectar tipo (manuscrita vs digital)
    variance = np.var(img)
    is_handwritten = variance > 800
    
    if debug:
        print(f"Tipo: {'Manuscrita' if is_handwritten else 'Digital'}")
    
    # Preprocesamiento según tipo
    if is_handwritten:
        # Manuscritas: más permisivo
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Digitales: estándar
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtros adaptativos
    boxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        
        if is_handwritten:
            # Filtros más permisivos para manuscritas
            min_area = 20
            min_dimension = 5
            max_width = w * 0.5
            max_height = h * 0.9
        else:
            # Filtros estrictos para digitales
            min_area = 50
            min_dimension = 8
            max_width = w * 0.3
            max_height = h * 0.8
        
        if (area >= min_area and cw >= min_dimension and ch >= min_dimension and
            cw <= max_width and ch <= max_height):
            boxes.append((x, y, cw, ch))
    
    if debug:
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for x, y, cw, ch in boxes:
            cv2.rectangle(debug_img, (x, y), (x+cw, y+ch), (0, 255, 0), 2)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.imshow(debug_img)
        plt.title(f'Segmentación mejorada - {len(boxes)} caracteres')
        plt.axis('off')
        plt.show()
    
    return boxes, img
