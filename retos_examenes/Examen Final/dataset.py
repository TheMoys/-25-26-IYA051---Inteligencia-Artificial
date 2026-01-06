import os
import numpy as np
from PIL import Image

# Mapeo de etiquetas numéricas a caracteres
LABEL_MAP = {
    **{i: str(i) for i in range(10)},  # 0-9 → '0'-'9'
    **{i + 10: chr(i + ord('A')) for i in range(26)},  # 10-35 → 'A'-'Z'
    **{i + 36: chr(i + ord('a')) for i in range(26)},  # 36-61 → 'a'-'z'
}

# Mapeo inverso: carácter a etiqueta numérica
CHAR_TO_LABEL = {
    **{str(i): i for i in range(10)},  # '0'-'9' → 0-9
    **{chr(i + ord('A')): i + 10 for i in range(26)},  # 'A'-'Z' → 10-35
    **{chr(i + ord('a')): i + 36 for i in range(26)},  # 'a'-'z' → 36-61
}

IMG_SIZE = (32, 32)


def extract_label_universal(filename):
    """
    Extrae la etiqueta del nombre del archivo detectando el formato automáticamente.
    
    Formatos soportados:
    1. DATASET_IA: "0_Nombre_Apellido.png" → primer carácter es la etiqueta
    2. DatasetCompleto2: "image_X_label_Y.png" → Y es la etiqueta numérica
    3. datasetCompleto: "img001-062.png" → formato secuencial antiguo
    """
    try:
        # FORMATO 1: DATASET_IA (0_Nombre.png, A_Nombre.png, a_Nombre.png)
        if filename[0] in CHAR_TO_LABEL and filename[1] == '_':
            char = filename[0]
            label = CHAR_TO_LABEL[char]
            return label
        
        # FORMATO 2: DatasetCompleto2 (image_X_label_Y.png)
        if "label_" in filename:
            parts = filename.split("label_")
            if len(parts) >= 2:
                label_str = parts[1].split('.')[0]  # Quitar extensión
                label = int(label_str)
                
                if 0 <= label <= 61:
                    return label
                else:
                    print(f"⚠️  Label fuera de rango ({label}): {filename}")
                    return None
        
        # FORMATO 3: datasetCompleto (img001-001.png, img001-062.png)
        if filename.startswith("img") and "-" in filename:
            # Extraer número después del guion
            parts = filename.split("-")
            if len(parts) >= 2:
                num_str = parts[1].split('.')[0]  # Quitar extensión
                img_index = int(num_str)
                
                # Mapeo: 001-010 → 0-9, 011-036 → A-Z, 037-062 → a-z
                if 1 <= img_index <= 10:
                    return img_index - 1  # 0-9
                elif 11 <= img_index <= 36:
                    return img_index - 11 + 10  # A-Z (10-35)
                elif 37 <= img_index <= 62:
                    return img_index - 37 + 36  # a-z (36-61)
        
        # FORMATO 3 alternativo: img001.png (sin guion)
        if filename.startswith("img") and filename[3:6].isdigit():
            img_index = int(filename[3:6])
            
            if 1 <= img_index <= 10:
                return img_index - 1
            elif 11 <= img_index <= 36:
                return img_index - 11 + 10
            elif 37 <= img_index <= 62:
                return img_index - 37 + 36
        
        print(f"⚠️  No se pudo extraer label de: {filename}")
        return None
        
    except Exception as e:
        print(f"❌ Error al procesar {filename}: {e}")
        return None


def load_dataset(dataset_path, dataset_type="new"):
    """
    Carga dataset detectando automáticamente el formato de nombres.
    
    Soporta:
    - DATASET_IA: 0_Nombre.png
    - DatasetCompleto2: image_X_label_Y.png
    - datasetCompleto: img001-062.png
    """
    images = []
    labels = []
    
    print(f"\n{'='*70}")
    print(f"📂 Cargando dataset desde: {dataset_path}")
    print(f"   Tipo especificado: {dataset_type}")
    print(f"{'='*70}")
    
    # Verificar que exista el directorio
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ No se encontró el dataset en: {dataset_path}")
    
    # Obtener todos los archivos de imagen
    all_files = sorted([f for f in os.listdir(dataset_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not all_files:
        raise ValueError(f"❌ No se encontraron imágenes en: {dataset_path}")
    
    print(f"📊 Total de archivos encontrados: {len(all_files)}")
    
    # Detectar formato automáticamente
    sample_file = all_files[0]
    print(f"🔍 Archivo de muestra: {sample_file}")
    
    if sample_file[0] in CHAR_TO_LABEL and sample_file[1] == '_':
        detected_format = "DATASET_IA (X_Nombre.png)"
    elif "label_" in sample_file:
        detected_format = "DatasetCompleto2 (image_X_label_Y.png)"
    elif sample_file.startswith("img"):
        detected_format = "datasetCompleto (imgXXX.png)"
    else:
        detected_format = "DESCONOCIDO"
    
    print(f"✨ Formato detectado: {detected_format}")
    
    # Estadísticas por clase
    label_counts = {}
    loaded_count = 0
    error_count = 0
    
    # Cargar cada imagen
    for img_name in all_files:
        img_path = os.path.join(dataset_path, img_name)
        
        # Extraer etiqueta del nombre (formato universal)
        label = extract_label_universal(img_name)
        
        if label is None:
            error_count += 1
            continue
        
        try:
            # Cargar y preprocesar imagen
            img = Image.open(img_path).convert("L")  # Escala de grises
            img = img.resize(IMG_SIZE)  # Redimensionar a 32x32
            
            # Convertir a array y normalizar
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            labels.append(label)
            
            # Contar por clase
            label_counts[label] = label_counts.get(label, 0) + 1
            loaded_count += 1
            
        except Exception as e:
            print(f"❌ Error al cargar {img_name}: {e}")
            error_count += 1
    
    # Mostrar estadísticas
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   ✅ Imágenes cargadas: {loaded_count}")
    print(f"   ❌ Errores: {error_count}")
    print(f"   📊 Clases únicas: {len(label_counts)}")
    
    # Mostrar distribución por clase (primeras 15)
    print(f"\n📊 Distribución de clases (muestra):")
    for label in sorted(label_counts.keys())[:15]:
        char = LABEL_MAP.get(label, "?")
        count = label_counts[label]
        print(f"   Label {label:2d} ('{char}'): {count:4d} imágenes")
    
    if len(label_counts) > 15:
        print(f"   ... y {len(label_counts) - 15} clases más")
    
    print(f"{'='*70}\n")
    
    # Convertir a arrays numpy
    if len(images) == 0:
        raise ValueError(f"❌ No se cargaron imágenes válidas desde {dataset_path}")
    
    images_array = np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    labels_array = np.array(labels)
    
    # Verificación final
    if len(images_array) != len(labels_array):
        raise ValueError(f"❌ Desbalance: {len(images_array)} imágenes vs {len(labels_array)} etiquetas")
    
    print(f"✅ Dataset cargado exitosamente:")
    print(f"   Shape de imágenes: {images_array.shape}")
    print(f"   Shape de etiquetas: {labels_array.shape}\n")
    
    return images_array, labels_array


# Funciones legacy (mantener por compatibilidad)
def extract_label_from_old_format(img_name):
    """Wrapper para compatibilidad"""
    return extract_label_universal(img_name)


def extract_label_from_new_format(img_name):
    """Wrapper para compatibilidad"""
    return extract_label_universal(img_name)