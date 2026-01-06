# diagnostico_labels.py
import os

# Verificar archivos de DatasetCompleto2
path = "./dataset/DatasetCompleto2"
files = sorted([f for f in os.listdir(path) if f.endswith('.png')])[:100]

print("=" * 70)
print("PRIMEROS 100 ARCHIVOS Y SUS LABELS")
print("=" * 70)

for i, filename in enumerate(files):
    # Extraer label del nombre
    if "label_" in filename:
        label_str = filename.split("label_")[1].split('.')[0]
        label = int(label_str)
        
        # Mostrar qué debería ser
        if label < 10:
            expected = str(label)  # Números 0-9
        elif 10 <= label <= 35:
            expected = chr(label - 10 + ord('A'))  # A-Z
        elif 36 <= label <= 61:
            expected = chr(label - 36 + ord('a'))  # a-z
        else:
            expected = "???"
        
        print(f"{i+1:3d}. {filename:40s} → Label {label:2d} → '{expected}'")
        
        if i < 10 or (60 < i < 70):  # Mostrar primeros 10 y algunos del medio
            continue
        elif i >= 70:
            break

print("\n" + "=" * 70)
print("VERIFICACIÓN:")
print("¿Los labels coinciden con lo que esperas?")
print("Ejemplo: image_X_label_0.png debería ser el número '0'")
print("         image_X_label_10.png debería ser la letra 'A'")
print("         image_X_label_36.png debería ser la letra 'a'")
print("=" * 70)