import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from model_training import train_model
from predict import predict_image, predict_folder


def train_interface():
    """
    Llama al entrenamiento manualmente a través de la interfaz gráfica.
    """
    train_model()


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

# Entrenamiento automático al inicio
print("Iniciando entrenamiento automático del modelo...")
train_model()
print("Entrenamiento completado.")

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("OCR Predictor")

button_letter = tk.Button(root, text="Predecir Letra", command=predict_letter)
button_letter.pack()

button_number = tk.Button(root, text="Predecir Número", command=predict_number)
button_number.pack()

button_phrase = tk.Button(root, text="Predecir Frase", command=predict_phrase)
button_phrase.pack()

button_folder = tk.Button(root, text="Predecir Carpeta", command=predict_from_folder)
button_folder.pack()

root.mainloop()
