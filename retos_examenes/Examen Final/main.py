import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import os
import sys
from model_training import train_model
from predict import predict_image, predict_folder, segment_and_predict_unified


def check_and_create_model():
    """
    Verifica si existe el modelo, si no lo entrena automáticamente.
    """
    if not os.path.exists("ocr_model.h5"):
        response = messagebox.askyesno(
            "Modelo no encontrado", 
            "No se encontró el modelo OCR. ¿Desea entrenar uno nuevo?\n\n" +
            "Esto puede tomar varios minutos."
        )
        if response:
            print("🚀 Iniciando entrenamiento automático...")
            try:
                train_model()
                messagebox.showinfo("Entrenamiento completo", "El modelo se ha entrenado exitosamente.")
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Error durante el entrenamiento:\n{str(e)}")
                return False
        else:
            messagebox.showwarning("Sin modelo", "No se puede usar la aplicación sin un modelo.")
            return False
    return True


def train_interface():
    """
    Entrena un nuevo modelo desde la interfaz.
    """
    response = messagebox.askyesno(
        "Re-entrenar modelo", 
        "¿Está seguro de que desea re-entrenar el modelo?\n\n" +
        "Esto sobrescribirá el modelo actual."
    )
    if response:
        try:
            print("🔄 Re-entrenando modelo...")
            train_model()
            messagebox.showinfo("Re-entrenamiento completo", "El modelo se ha re-entrenado exitosamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el re-entrenamiento:\n{str(e)}")


def debug_interface():
    """
    Debug de segmentación.
    """
    if not check_and_create_model():
        return
        
    image_path = filedialog.askopenfilename(
        title="Selecciona una imagen para debug",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if image_path:
        print(f"🔍 Analizando: {image_path}")
        try:
            segment_and_predict_unified(image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error en debug:\n{str(e)}")


def predict_letter():
    """
    Predice una letra individual.
    """
    if not check_and_create_model():
        return
        
    image_path = filedialog.askopenfilename(
        title="Selecciona imagen de una letra",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if image_path:
        try:
            prediction = predict_image(image_path, "letter")
            print(f"Predicción (Letra): {prediction}")
            messagebox.showinfo("Resultado", f"Letra predicha: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción:\n{str(e)}")


def predict_number():
    """
    Predice un número individual.
    """
    if not check_and_create_model():
        return
        
    image_path = filedialog.askopenfilename(
        title="Selecciona imagen de un número",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if image_path:
        try:
            prediction = predict_image(image_path, "number")
            print(f"Predicción (Número): {prediction}")
            messagebox.showinfo("Resultado", f"Número predicho: {prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción:\n{str(e)}")


def predict_phrase():
    """
    Predice una frase completa.
    """
    if not check_and_create_model():
        return
        
    image_path = filedialog.askopenfilename(
        title="Selecciona imagen con texto",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if image_path:
        try:
            prediction, _ = segment_and_predict_unified(image_path)
            print(f"Predicción (Frase): {prediction}")
            messagebox.showinfo("Resultado", f"Texto predicho:\n\n{prediction}")
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción:\n{str(e)}")


def predict_from_folder():
    """
    Predice múltiples imágenes de una carpeta.
    """
    if not check_and_create_model():
        return
        
    folder_path = filedialog.askdirectory(title="Selecciona la carpeta de imágenes")
    if folder_path:
        try:
            model = tf.keras.models.load_model("ocr_model.h5")
            accuracy = predict_folder(folder_path, model)
            if accuracy > 0:
                print(f"Precisión calculada: {accuracy:.2f}%")
                messagebox.showinfo("Resultado", f"Procesamiento completo.\nPrecisión: {accuracy:.2f}%")
            else:
                messagebox.showwarning("Sin resultados", "No se procesaron imágenes válidas.")
        except Exception as e:
            messagebox.showerror("Error", f"Error procesando carpeta:\n{str(e)}")


def main():
    """
    Función principal que maneja tanto GUI como línea de comandos.
    """
    # Si hay argumentos de línea de comandos, usar modo consola
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            print("🚀 Iniciando entrenamiento...")
            train_model()
            
        elif command == "predict" and len(sys.argv) >= 3:
            image_path = sys.argv[2]
            if os.path.exists(image_path):
                print(f"🔍 Prediciendo: {image_path}")
                phrase, _ = segment_and_predict_unified(image_path)
                print(f"\n✅ RESULTADO FINAL: '{phrase}'")
            else:
                print(f"❌ No existe el archivo: {image_path}")
                
        elif command == "predict_folder" and len(sys.argv) >= 3:
            folder_path = sys.argv[2]
            if os.path.exists(folder_path):
                try:
                    model = tf.keras.models.load_model("ocr_model.h5")
                    accuracy = predict_folder(folder_path, model)
                    print(f"\n✅ Precisión final: {accuracy:.2f}%")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print(f"❌ No existe la carpeta: {folder_path}")
                
        else:
            print("Uso:")
            print("  python main.py train                     - Entrenar modelo")
            print("  python main.py predict <imagen>          - Predecir imagen")
            print("  python main.py predict_folder <carpeta>  - Predecir carpeta")
        
        return
    
    # Modo GUI
    root = tk.Tk()
    root.title("OCR Predictor - Actualizado")
    root.geometry("350x400")
    root.resizable(False, False)
    
    # Título
    title_label = tk.Label(root, text="🔤 OCR Predictor", font=("Arial", 16, "bold"), fg="darkblue")
    title_label.pack(pady=10)
    
    # Verificar modelo al inicio
    if not os.path.exists("ocr_model.h5"):
        status_label = tk.Label(root, text="⚠️ Modelo no encontrado", fg="red")
        status_label.pack()
    else:
        status_label = tk.Label(root, text="✅ Modelo cargado", fg="green")
        status_label.pack()
    
    # Botón de DEBUG destacado
    button_debug = tk.Button(root, text="🔍 Debug Segmentación", command=debug_interface, 
                           bg="lightblue", font=("Arial", 11, "bold"), width=25)
    button_debug.pack(pady=10)
    
    # Separador
    separator1 = tk.Label(root, text="─" * 35, fg="gray")
    separator1.pack()
    
    # Botón de entrenamiento
    button_train = tk.Button(root, text="🔄 Re-entrenar Modelo", command=train_interface, 
                           bg="orange", font=("Arial", 10), width=25)
    button_train.pack(pady=5)
    
    # Separador
    separator2 = tk.Label(root, text="─" * 35, fg="gray")
    separator2.pack()
    
    # Etiqueta de predicciones
    pred_label = tk.Label(root, text="Predicciones:", font=("Arial", 12, "bold"))
    pred_label.pack(pady=(10, 5))
    
    # Botones de predicción
    button_letter = tk.Button(root, text="📝 Predecir Letra", command=predict_letter, 
                            width=25, font=("Arial", 10))
    button_letter.pack(pady=3)
    
    button_number = tk.Button(root, text="🔢 Predecir Número", command=predict_number, 
                            width=25, font=("Arial", 10))
    button_number.pack(pady=3)
    
    button_phrase = tk.Button(root, text="📄 Predecir Frase", command=predict_phrase, 
                            width=25, font=("Arial", 10), bg="lightgreen")
    button_phrase.pack(pady=3)
    
    button_folder = tk.Button(root, text="📁 Predecir Carpeta", command=predict_from_folder, 
                            width=25, font=("Arial", 10))
    button_folder.pack(pady=3)
    
    # Información adicional
    info_label = tk.Label(root, text="Tip: Usa 'Debug Segmentación' para\nanalizar cómo procesa las imágenes", 
                         font=("Arial", 9), fg="gray", justify="center")
    info_label.pack(pady=15)
    
    root.mainloop()


if __name__ == "__main__":
    main()