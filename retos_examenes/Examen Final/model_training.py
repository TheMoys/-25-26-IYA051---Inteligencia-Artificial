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
    print("Cargando dataset de entrenamiento 1...")
    train_images1, train_labels1 = load_dataset("./dataset/DatasetCompleto2", dataset_type="new")
    print(f"  ✓ Dataset 1: {train_images1.shape[0]} imágenes, {train_labels1.shape[0]} etiquetas")

    print("Cargando dataset de entrenamiento 2...")
    train_images2, train_labels2 = load_dataset("./dataset/DATASET_IA", dataset_type="new")
    print(f"  ✓ Dataset 2: {train_images2.shape[0]} imágenes, {train_labels2.shape[0]} etiquetas")

    print("Cargando dataset de validación...")
    val_images, val_labels = load_dataset("./dataset/datasetCompleto", dataset_type="old")
    print(f"  ✓ Validación: {val_images.shape[0]} imágenes, {val_labels.shape[0]} etiquetas")

    # VERIFICAR CONSISTENCIA de cada dataset
    if train_images1.shape[0] != train_labels1.shape[0]:
        print(f"⚠️  ERROR: Dataset 1 desbalanceado!")
        print(f"   Imágenes: {train_images1.shape[0]}, Etiquetas: {train_labels1.shape[0]}")
        # Recortar al tamaño mínimo
        min_size1 = min(train_images1.shape[0], train_labels1.shape[0])
        train_images1 = train_images1[:min_size1]
        train_labels1 = train_labels1[:min_size1]
        print(f"   Recortado a: {min_size1}")

    if train_images2.shape[0] != train_labels2.shape[0]:
        print(f"⚠️  ERROR: Dataset 2 desbalanceado!")
        print(f"   Imágenes: {train_images2.shape[0]}, Etiquetas: {train_labels2.shape[0]}")
        min_size2 = min(train_images2.shape[0], train_labels2.shape[0])
        train_images2 = train_images2[:min_size2]
        train_labels2 = train_labels2[:min_size2]
        print(f"   Recortado a: {min_size2}")

    if val_images.shape[0] != val_labels.shape[0]:
        print(f"⚠️  ERROR: Dataset de validación desbalanceado!")
        min_size_val = min(val_images.shape[0], val_labels.shape[0])
        val_images = val_images[:min_size_val]
        val_labels = val_labels[:min_size_val]

    # COMBINAR los datasets de entrenamiento
    print("\nCombinando datasets de entrenamiento...")
    train_images = np.concatenate([train_images1, train_images2], axis=0)
    train_labels = np.concatenate([train_labels1, train_labels2], axis=0)

    print(f"  Dataset combinado: {train_images.shape[0]} imágenes, {train_labels.shape[0]} etiquetas")

    # VERIFICAR después de combinar
    if train_images.shape[0] != train_labels.shape[0]:
        print(f"❌ ERROR CRÍTICO: Tamaños no coinciden después de combinar!")
        print(f"   Imágenes: {train_images.shape[0]}")
        print(f"   Etiquetas: {train_labels.shape[0]}")
        return None

    # Mezclar aleatoriamente
    print("Mezclando aleatoriamente...")
    shuffle_indices = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    # Verificar datos cargados
    print(f"\n📊 RESUMEN FINAL:")
    print(f"   Dataset 1: {train_images1.shape}")
    print(f"   Dataset 2: {train_images2.shape}")
    print(f"   Dataset combinado: {train_images.shape}, Etiquetas: {train_labels.shape}")
    print(f"   Dataset de validación: {val_images.shape}, Etiquetas: {val_labels.shape}")

    # Aplicar augmentación de datos
    print("\nAplicando augmentación de datos al dataset de entrenamiento...")
    train_images = augment_images(train_images)
    print(f"   Dataset después de augmentación: {train_images.shape}")

    # Crear el modelo con Dropout y L2
    model = create_optimized_model((32, 32, 1), 62, use_dropout=True, l2_reg=0.001)

    # Definir Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Entrenar el modelo
    print("\n🚀 INICIANDO ENTRENAMIENTO...")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Guardar el modelo entrenado
    model.save("ocr_model.h5")
    print("\n✅ Modelo guardado como 'ocr_model.h5'")

    # Calcular precisión
    print("\nCalculando precisión en el conjunto de validación...")
    val_predictions = model.predict(val_images)
    val_predicted_classes = np.argmax(val_predictions, axis=1)

    correct_predictions = np.sum(val_predicted_classes == val_labels)
    total_predictions = len(val_labels)
    accuracy = (correct_predictions / total_predictions) * 100

    print(f"\n📊 Precisión en el conjunto de validación: {accuracy:.2f}%")

    return history
