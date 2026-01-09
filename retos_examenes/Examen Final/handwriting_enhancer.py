import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_handwriting(char_img, debug=False):
    """
    Preprocesamiento especializado para caracteres manuscritos.
    """
    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    
    original = char_img.copy()
    
    # 1. Corrección de iluminación para manuscritas
    kernel_size = max(21, min(char_img.shape) // 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Estimar fondo
    background = cv2.GaussianBlur(char_img, (kernel_size, kernel_size), 0)
    corrected = cv2.divide(char_img, background, scale=255)
    
    # 2. Mejora de contraste agresiva para manuscritas
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(corrected)
    
    # 3. Detección automática de inversión
    mean_val = np.mean(enhanced)
    if mean_val > 127:  # Fondo claro, invertir
        enhanced = 255 - enhanced
    
    # 4. Suavizado para conectar trazos manuscritos rotos
    kernel_smooth = np.ones((2, 2), np.uint8)
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_smooth)
    
    # 5. Binarización adaptativa múltiple (probar diferentes métodos)
    methods = []
    
    # Método 1: Otsu estándar
    _, bin1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    methods.append(("Otsu", bin1))
    
    # Método 2: Adaptativo Gaussiano
    bin2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 15, 8)
    methods.append(("Gaussian", bin2))
    
    # Método 3: Adaptativo promedio
    bin3 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 15, 8)
    methods.append(("Mean", bin3))
    
    # 6. Seleccionar el mejor método basado en cantidad de información
    best_method = None
    best_score = 0
    best_binary = None
    
    for name, binary in methods:
        # Calcular score: balance entre información y ruido
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        ratio = white_pixels / total_pixels
        
        # Preferir ratio entre 0.1 y 0.4 (texto claro sobre fondo)
        if 0.1 <= ratio <= 0.4:
            score = 1.0 - abs(ratio - 0.25)  # Óptimo en 0.25
        else:
            score = 0
        
        if score > best_score:
            best_score = score
            best_method = name
            best_binary = binary
    
    if best_binary is None:
        best_binary = bin1  # Fallback a Otsu
        best_method = "Otsu (fallback)"
    
    # 7. Limpiar ruido pequeño
    kernel_clean = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(best_binary, cv2.MORPH_OPEN, kernel_clean)
    
    # 8. Encontrar y extraer el carácter principal
    coords = np.column_stack(np.where(cleaned > 0))
    if len(coords) == 0:
        # Si no hay contenido, devolver imagen vacía
        result = np.zeros((32, 32), dtype=np.float32)
        if debug:
            print("⚠️ Sin contenido después del procesamiento")
        return result
    
    # Bounding box apretado
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    char_tight = cleaned[y_min:y_max+1, x_min:x_max+1]
    
    # 9. Redimensionar con padding inteligente
    h, w = char_tight.shape
    max_dim = max(h, w)
    
    # Padding del 25% para manuscritas (más generoso)
    padded_size = int(max_dim * 1.25)
    
    # Centrar
    pad_h = (padded_size - h) // 2
    pad_w = (padded_size - w) // 2
    
    padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
    padded[pad_h:pad_h+h, pad_w:pad_w+w] = char_tight
    
    # 10. Redimensionar a 32x32 con interpolación suave
    resized = cv2.resize(padded, (32, 32), interpolation=cv2.INTER_AREA)
    
    # 11. Normalización final
    normalized = resized / 255.0
    
    # 12. Aplicar ligero engrosado para manuscritas débiles
    kernel_thick = np.ones((2, 2), np.float32) / 4
    thickened = cv2.filter2D(normalized, -1, kernel_thick)
    
    # Umbralizar de nuevo
    final = np.where(thickened > 0.15, 1.0, 0.0)  # Umbral bajo para manuscritas
    
    if debug:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title('Original')
        axes[0,1].imshow(enhanced, cmap='gray')
        axes[0,1].set_title('Enhanced')
        axes[0,2].imshow(best_binary, cmap='gray')
        axes[0,2].set_title(f'Best Binary ({best_method})')
        axes[0,3].imshow(char_tight, cmap='gray')
        axes[0,3].set_title('Tight Crop')
        axes[1,0].imshow(padded, cmap='gray')
        axes[1,0].set_title('Padded')
        axes[1,1].imshow(resized, cmap='gray')
        axes[1,1].set_title('Resized')
        axes[1,2].imshow(thickened, cmap='gray')
        axes[1,2].set_title('Thickened')
        axes[1,3].imshow(final, cmap='gray')
        axes[1,3].set_title('Final 32x32')
        
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return final.astype(np.float32)