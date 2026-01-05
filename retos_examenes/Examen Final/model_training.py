import numpy as np
import tensorflow as tf
from dataset import load_dataset
from model import create_optimized_model

def rotate_image(img, angle):
    """
    Rota una imagen manualmente utilizando TensorFlow.
    """
    angle = tf.constant(angle, dtype=tf.float32)
    return tf.image.rot90(img, k=int(angle // (np.pi / 2)))


def augment_images(images):
    """
    Aplica augmentación de datos básica utilizando solo TensorFlow.
    """
    augmented_images = []
    for img in images:
        img = tf.image.random_flip_left_right(img)  # Volteo horizontal
        img = tf.image.random_brightness(img, max_delta=0.1)  # Ajuste de brillo
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)  # Ajuste de contraste
        img = rotate_image(img, np.pi / 4)  # Rotación de 45 grados
        augmented_images.append(img.numpy())  # Convertir a numpy para evitar errores
    return np.array(augmented_images)


def train_model():
    # Cargar datasets
    print("Cargando dataset de entrenamiento...")
    train_images, train_labels = load_dataset("./dataset/DatasetCompleto2", dataset_type="new")

    print("Cargando dataset de validación...")
    val_images, val_labels = load_dataset("./dataset/datasetCompleto", dataset_type="old")

    # Verificar datos cargados
    print(f"Dataset de entrenamiento: {train_images.shape}, {train_labels.shape}")
    print(f"Dataset de validación: {val_images.shape}, {val_labels.shape}")

    # Aplicar augmentación de datos
    print("Aplicando augmentación de datos al dataset de entrenamiento...")
    train_images = augment_images(train_images)

    # Crear el modelo con Dropout y L2
    model = create_optimized_model((32, 32, 1), 62, use_dropout=True, l2_reg=0.001)

    # Definir Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,  # Detener después de 5 épocas sin mejora
        restore_best_weights=True
    )

    # Compilar el modelo con un learning rate menor para más estabilidad
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduce el learning rate
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Entrenar el modelo con callbacks
    print("Entrenando el modelo con regularización y Early Stopping...")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=10,  # Máximo de 10 épocas, Early Stopping detendrá antes
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Guardar el modelo entrenado
    model.save("ocr_model.h5")
    print("Modelo guardado.")

        # Calcular precisión en el conjunto de validación
    print("Calculando precisión en el conjunto de validación...")
    val_predictions = model.predict(val_images)
    val_predicted_classes = np.argmax(val_predictions, axis=1)

    # Comparar con etiquetas reales
    correct_predictions = np.sum(val_predicted_classes == val_labels)
    total_predictions = len(val_labels)
    accuracy = (correct_predictions / total_predictions) * 100

    print(f"Precisión en el conjunto de validación: {accuracy:.2f}%")

    # Retornar historial para análisis
    return history
    
