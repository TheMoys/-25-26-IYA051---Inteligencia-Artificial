import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
import os
import sys
from model_training import train_model
from predict import predict_universal 


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


def predict_universal_gui():
    """
    NUEVA: Función universal que detecta automáticamente el tipo de contenido.
    """
    if not check_and_create_model():
        return
        
    image_path = filedialog.askopenfilename(
        title="Selecciona una imagen con texto (letra, palabra o frase)",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff")]
    )
    if image_path:
        try:
            print(f"🔍 Procesando: {image_path}")
            
            # Preguntar si quiere modo debug
            debug_response = messagebox.askyesno(
                "Modo Debug", 
                "¿Activar modo debug para ver detalles del procesamiento?"
            )
            
            # Usar la nueva función universal
            prediction, boxes = predict_universal(image_path, debug=debug_response)
            
            print(f"✅ Resultado: '{prediction}'")
            
            # Mostrar resultado con información del tipo detectado
            num_chars = len(boxes)
            if num_chars == 1:
                content_type = "LETRA INDIVIDUAL"
                emoji = "🔤"
            elif 2 <= num_chars <= 6:
                content_type = "PALABRA"
                emoji = "📝"
            else:
                content_type = "FRASE"
                emoji = "📄"
            
            # Ventana de resultado mejorada
            result_message = f"{emoji} Tipo detectado: {content_type}\n"
            result_message += f"🔍 Caracteres encontrados: {num_chars}\n"
            result_message += f"📝 Texto reconocido:\n\n'{prediction}'"
            
            messagebox.showinfo("Resultado del Reconocimiento", result_message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en el reconocimiento:\n{str(e)}")
            print(f"❌ Error: {e}")


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
                phrase, _ = predict_universal(image_path, debug=True)
                print(f"\n✅ RESULTADO FINAL: '{phrase}'")
            else:
                print(f"❌ No existe el archivo: {image_path}")
                
        else:
            print("Uso:")
            print("  python main.py train                - Entrenar modelo")
            print("  python main.py predict <imagen>     - Predecir imagen")
        
        return
    
    # NUEVA GUI SIMPLIFICADA
    root = tk.Tk()
    root.title("OCR Universal - Sistema Inteligente")
    root.geometry("400x350")
    root.resizable(False, False)
    
    # Configurar colores y estilo
    root.configure(bg='#f0f0f0')
    
    # Título principal
    title_label = tk.Label(root, text="🔤 OCR UNIVERSAL", 
                          font=("Arial", 18, "bold"), 
                          fg="#2c3e50", bg='#f0f0f0')
    title_label.pack(pady=15)
    
    # Subtítulo descriptivo
    subtitle_label = tk.Label(root, text="Reconocimiento automático de texto\n✨ Detecta letras, palabras y frases", 
                             font=("Arial", 11), 
                             fg="#34495e", bg='#f0f0f0')
    subtitle_label.pack(pady=5)
    
    # Estado del modelo
    if not os.path.exists("ocr_model.h5"):
        status_label = tk.Label(root, text="⚠️ Modelo no encontrado", 
                               fg="red", bg='#f0f0f0', font=("Arial", 10))
        status_label.pack()
    else:
        status_label = tk.Label(root, text="✅ Modelo listo", 
                               fg="green", bg='#f0f0f0', font=("Arial", 10))
        status_label.pack()
    
    # Separador
    separator1 = tk.Frame(root, height=2, bg="#bdc3c7")
    separator1.pack(fill=tk.X, padx=30, pady=15)
    
    # BOTÓN PRINCIPAL - RECONOCIMIENTO UNIVERSAL
    button_predict = tk.Button(root, text="🎯 RECONOCER TEXTO", 
                              command=predict_universal_gui,
                              bg="#3498db", fg="white", 
                              font=("Arial", 14, "bold"), 
                              width=25, height=2,
                              relief=tk.RAISED, bd=3)
    button_predict.pack(pady=15)
    
    # Descripción del botón principal
    desc_label = tk.Label(root, text="Detecta automáticamente:\n🔤 Letras individuales  📝 Palabras  📄 Frases\n✍️ Texto manuscrito y digital", 
                         font=("Arial", 9), 
                         fg="#7f8c8d", bg='#f0f0f0', justify="center")
    desc_label.pack(pady=10)
    
    # Separador
    separator2 = tk.Frame(root, height=1, bg="#ecf0f1")
    separator2.pack(fill=tk.X, padx=50, pady=10)
    
    # Botón de entrenamiento (secundario)
    button_train = tk.Button(root, text="🔄 Re-entrenar Modelo", 
                           command=train_interface,
                           bg="#e67e22", fg="white", 
                           font=("Arial", 10), 
                           width=20, height=1)
    button_train.pack(pady=5)
    
    # Información adicional
    info_frame = tk.Frame(root, bg='#f0f0f0')
    info_frame.pack(pady=15)
    
    info_label = tk.Label(info_frame, 
                         text="💡 Tip: La función universal analiza automáticamente\nel contenido y aplica el procesamiento óptimo", 
                         font=("Arial", 9), 
                         fg="#95a5a6", bg='#f0f0f0', 
                         justify="center")
    info_label.pack()
    
    # Footer
    footer_label = tk.Label(root, text="v2.0 - Sistema OCR Inteligente", 
                           font=("Arial", 8), 
                           fg="#bdc3c7", bg='#f0f0f0')
    footer_label.pack(side=tk.BOTTOM, pady=5)
    
    root.mainloop()


if __name__ == "__main__":
    main()