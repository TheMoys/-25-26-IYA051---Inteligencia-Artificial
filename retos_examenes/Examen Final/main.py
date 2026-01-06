import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import os
from model_training import train_model
from predict import predict_image, predict_folder, debug_segmentation


def train_interface():
    """
    Llama al entrenamiento manualmente a través de la interfaz gráfica.
    """
    train_model()


def debug_interface():
    """
    Muestra el proceso de segmentación paso a paso.
    """
    image_path = filedialog.askopenfilename(title="Selecciona una imagen para debug")
    if image_path:
        print(f"🔍 Analizando: {image_path}")
        debug_segmentation(image_path)


def predict_letter():
    image_path = filedialog.askopenfilename(title="Selecciona una imagen")
    if image_path:
        prediction = predict_image(image_path, "letter")
        print(f"Predicción (Letra): {prediction}")


def predict_number():
    image_path = filedialog.askopenfilename(title="Selecciona una imagen")
    if image_path:
        prediction = predict_image(image_path, "number")
        print(f"Predicción (Número): {prediction}")


def predict_phrase():
    image_path = filedialog.askopenfilename(title="Selecciona una imagen")
    if image_path:
        prediction = predict_image(image_path, "phrase")
        print(f"Predicción (Frase): {prediction}")


def predict_from_folder():
    model = tf.keras.models.load_model("ocr_model.h5")  # Cargar el modelo
    folder_path = filedialog.askdirectory(title="Selecciona la carpeta de imágenes")
    if folder_path:
        accuracy = predict_folder(folder_path, model)
        if accuracy > 0:
            print(f"Precisión calculada: {accuracy:.2f}%")
        else:
            print("No se procesaron imágenes válidas o precisión es 0%.")


# Verificar si el modelo existe antes de entrenar
if not os.path.exists("ocr_model.h5"):
    print("⚠️  Modelo no encontrado. Iniciando entrenamiento automático...")
    train_model()
    print("✅ Entrenamiento completado y modelo guardado.")
else:
    print("✅ Modelo encontrado. Cargando modelo existente...")
    # No hacer nada, el modelo ya existe

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("OCR Predictor")
root.geometry("300x350")  # Tamaño de ventana más grande

# Botón de DEBUG (nuevo) - en color azul para destacarlo
button_debug = tk.Button(root, text="🔍 Debug Segmentación", command=debug_interface, bg="lightblue", font=("Arial", 10, "bold"))
button_debug.pack(pady=10)

# Separador visual
separator = tk.Label(root, text="─" * 30)
separator.pack()

# Botón para re-entrenar manualmente si es necesario
button_train = tk.Button(root, text="🔄 Re-entrenar Modelo", command=train_interface, bg="orange")
button_train.pack(pady=5)

# Botones de predicción
button_letter = tk.Button(root, text="Predecir Letra", command=predict_letter, width=20)
button_letter.pack(pady=3)

button_number = tk.Button(root, text="Predecir Número", command=predict_number, width=20)
button_number.pack(pady=3)

button_phrase = tk.Button(root, text="Predecir Frase", command=predict_phrase, width=20)
button_phrase.pack(pady=3)

button_folder = tk.Button(root, text="Predecir Carpeta", command=predict_from_folder, width=20)
button_folder.pack(pady=3)

root.mainloop()