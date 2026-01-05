import os
import numpy as np
from PIL import Image

LABEL_MAP = {
    **{i: str(i) for i in range(10)},  # Números: 0-9
    **{i + 10: chr(i + ord('A')) for i in range(26)},  # Mayúsculas: A-Z
    **{i + 36: chr(i + ord('a')) for i in range(26)},  # Minúsculas: a-z
}

IMG_SIZE = (32, 32)  # Tamaño al que se redimensionarán las imágenes

def extract_label_from_old_format(img_name):
    try:
        if img_name.startswith("img"):
            img_index = int(img_name[3:6])
            if 1 <= img_index <= 10:
                return img_index - 1
            elif 11 <= img_index <= 36:
                return img_index - 11 + 10
            elif 37 <= img_index <= 62:
                return img_index - 37 + 36
            else:
                raise ValueError(f"Índice fuera de rango: {img_index}")
        else:
            raise ValueError(f"Formato inesperado: {img_name}")
    except Exception as e:
        print(f"Error al procesar el archivo {img_name}: {e}")
        return None

def extract_label_from_new_format(img_name):
    try:
        numeric_label = int(img_name.split('_')[-1].split('.')[0])
        return numeric_label
    except ValueError:
        raise ValueError(f"Error al procesar el nombre del archivo: {img_name}")

def load_dataset(dataset_path, dataset_type="new"):
    images = []
    labels = []

    print(f"Cargando dataset desde: {dataset_path}, Tipo: {dataset_type}")
    for img_name in sorted(os.listdir(dataset_path)):
        img_path = os.path.join(dataset_path, img_name)
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize(IMG_SIZE)  # Redimensionar a 32x32
            images.append(np.array(img, dtype=np.float32) / 255.0)

            if dataset_type == "old":
                label = extract_label_from_old_format(img_name)
            elif dataset_type == "new":
                label = extract_label_from_new_format(img_name)
            else:
                raise ValueError("Tipo de dataset desconocido.")

            if label is not None:
                labels.append(label)
                print(f"Imagen: {img_name}, Etiqueta asignada: {LABEL_MAP[label]}")
        except Exception as e:
            print(f"Error al procesar el archivo {img_name}: {e}")

    return np.array(images).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1), np.array(labels)
