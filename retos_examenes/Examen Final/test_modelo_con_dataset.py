# test_modelo_con_dataset.py
import cv2
import numpy as np
import tensorflow as tf
from dataset import LABEL_MAP
import matplotlib.pyplot as plt

# Cargar modelo
model = tf.keras.models.load_model("ocr_model.h5")

# Probar con 10 imágenes del dataset de entrenamiento
import os

dataset_path = "./dataset/DatasetCompleto2"
test_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.png')])[:100:10]  # Cada 10

print("=" * 80)
print("PROBANDO MODELO CON IMÁGENES DEL DATASET DE ENTRENAMIENTO")
print("=" * 80)

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

correct = 0
total = 0

for i, filename in enumerate(test_files):
    img_path = os.path.join(dataset_path, filename)
    
    # Extraer label del nombre
    if "label_" in filename:
        true_label = int(filename.split("label_")[1].split('.')[0])
        true_char = LABEL_MAP[true_label]
    else:
        print(f"Saltando {filename} (sin label)")
        continue
    
    # Cargar imagen EXACTAMENTE como en dataset.py
    from PIL import Image
    img = Image.open(img_path).convert("L")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_input = img_array.reshape(1, 32, 32, 1)
    
    # Predecir
    prediction = model.predict(img_input, verbose=0)
    predicted_class = np.argmax(prediction)
    predicted_char = LABEL_MAP.get(predicted_class, "?")
    confidence = np.max(prediction)
    
    # Comparar
    is_correct = (predicted_class == true_label)
    if is_correct:
        correct += 1
    total += 1
    
    # Visualizar
    axes[i].imshow(img_array.reshape(32, 32), cmap='gray')
    color = 'green' if is_correct else 'red'
    axes[i].set_title(f"Real: {true_char} | Pred: {predicted_char}\nConf: {confidence:.2f}", 
                     fontsize=10, color=color, fontweight='bold')
    axes[i].axis('off')
    
    # Log
    status = "✅" if is_correct else "❌"
    print(f"{status} {filename:40s} | Real: {true_char:3s} (label {true_label:2d}) | "
          f"Pred: {predicted_char:3s} (label {predicted_class:2d}) | Conf: {confidence:.2%}")

plt.tight_layout()
plt.savefig("test_dataset_results.png", dpi=150)
plt.show()

accuracy = (correct / total) * 100 if total > 0 else 0
print("\n" + "=" * 80)
print(f"📊 RESULTADO: {correct}/{total} correctas = {accuracy:.2f}% accuracy")
print("=" * 80)

if accuracy < 50:
    print("\n❌ PROBLEMA CRÍTICO: El modelo no reconoce ni las imágenes del dataset")
    print("   Posibles causas:")
    print("   1. Las etiquetas están mal asignadas en el dataset")
    print("   2. El modelo no se entrenó correctamente")
    print("   3. Hay un bug en load_dataset() o en el mapeo de clases")
else:
    print("\n✅ El modelo funciona bien con el dataset")
    print("   El problema es el preprocesamiento de imágenes nuevas")