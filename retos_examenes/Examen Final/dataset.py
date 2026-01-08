import os
import cv2
import numpy as np
from PIL import Image

# Mapeo CORRECTO y UNIFICADO
LABEL_MAP = {
    # Números 0-9 (labels 0-9)
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    # Mayúsculas A-Z (labels 10-35)
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    # Minúsculas a-z (labels 36-61)
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

CHAR_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

IMG_SIZE = (32, 32)


def preprocess_char_unified(img_array):
    """
    Preprocesamiento UNIFICADO para caracteres individuales.
    Esta función se usa TANTO en entrenamiento como en predicción.
    """
    # Asegurar que es escala de grises
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar a 32x32
    img_resized = cv2.resize(img_array, IMG_SIZE)
    
    # Detectar si es fondo blanco o negro
    mean_intensity = np.mean(img_resized)
    
    if mean_intensity > 127:  # Fondo blanco, texto negro
        # Invertir para que sea fondo negro, texto blanco
        img_normalized = (255 - img_resized) / 255.0
    else:  # Ya está con fondo negro, texto blanco
        img_normalized = img_resized / 255.0
    
    # Aplicar umbralización para limpiar
    img_normalized = np.where(img_normalized > 0.3, 1.0, 0.0)
    
    return img_normalized.astype(np.float32)


def extract_label_universal(filename):
    """
    Extrae etiqueta del nombre de archivo - TODOS LOS FORMATOS
    """
    try:
        # FORMATO 1: DATASET_IA (0_Nombre.png, A_Nombre.png, a_Nombre.png)
        if len(filename) > 1 and filename[1] == '_':
            char = filename[0]
            if char in CHAR_TO_LABEL:
                return CHAR_TO_LABEL[char]
        
        # FORMATO 2: DatasetCompleto2 (image_X_label_Y.png)
        if "label_" in filename:
            parts = filename.split("label_")
            if len(parts) >= 2:
                label_str = parts[1].split('.')[0]
                label = int(label_str)
                if 0 <= label <= 61:
                    return label
        
        # FORMATO 3: datasetCompleto (img001-001.png a img001-062.png)
        if filename.startswith("img") and "-" in filename:
            parts = filename.split("-")
            if len(parts) >= 2:
                num_str = parts[1].split('.')[0]
                img_index = int(num_str)
                # Mapeo: 1-10 → 0-9, 11-36 → A-Z (10-35), 37-62 → a-z (36-61)
                if 1 <= img_index <= 10:
                    return img_index - 1
                elif 11 <= img_index <= 36:
                    return img_index - 11 + 10
                elif 37 <= img_index <= 62:
                    return img_index - 37 + 36
        
        return None
    except:
        return None


def load_dataset(dataset_path, dataset_type="new"):
    """
    Carga dataset con preprocesamiento UNIFICADO.
    """
    images = []
    labels = []
    
    print(f"\n📂 Cargando: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No existe: {dataset_path}")
    
    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not files:
        raise ValueError(f"No hay imágenes en: {dataset_path}")
    
    print(f"   📊 {len(files)} archivos encontrados")
    
    loaded_count = 0
    error_count = 0
    
    for filename in files:
        filepath = os.path.join(dataset_path, filename)
        
        # Extraer label
        label = extract_label_universal(filename)
        if label is None:
            error_count += 1
            continue
        
        try:
            # Cargar imagen
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                error_count += 1
                continue
            
            # PREPROCESAR con función UNIFICADA
            img_processed = preprocess_char_unified(img)
            
            images.append(img_processed)
            labels.append(label)
            loaded_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    print(f"   ✅ Cargadas: {loaded_count}")
    print(f"   ❌ Errores: {error_count}")
    
    # Convertir a arrays
    images_array = np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    labels_array = np.array(labels)
    
    print(f"   📊 Shape final: {images_array.shape}")
    
    return images_array, labels_array


# Funciones legacy para compatibilidad
def extract_label_from_old_format(img_name):
    return extract_label_universal(img_name)

def extract_label_from_new_format(img_name):
    return extract_label_universal(img_name)