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
    # Cargar SOLO datasets limpios para entrenamiento
    print("Cargando dataset de entrenamiento 1 (tipografías digitales)...")
    train_images1, train_labels1 = load_dataset("./dataset/DatasetCompleto2", dataset_type="new")
    print(f"  ✓ Dataset 1: {train_images1.shape[0]} imágenes")

    print("Cargando dataset de entrenamiento 2 (manuscritas digitalizadas)...")
    train_images2, train_labels2 = load_dataset("./dataset/datasetCompleto", dataset_type="old")
    print(f"  ✓ Dataset 2: {train_images2.shape[0]} imágenes")

    # NO USAR DATASET_IA para entrenamiento (tiene fotos sin preprocesar)
    # print("Cargando DATASET_IA...")
    # train_images3, train_labels3 = load_dataset("./dataset/DATASET_IA", dataset_type="new")
    
    print("Cargando dataset de validación...")
    # Usar parte de datasetCompleto para validación
    val_images, val_labels = load_dataset("./dataset/datasetCompleto", dataset_type="old")
    
    # Dividir datasetCompleto: 80% entrenamiento, 20% validación
    split_idx = int(len(train_images2) * 0.8)
    train_images2_train = train_images2[:split_idx]
    train_labels2_train = train_labels2[:split_idx]
    val_images = train_images2[split_idx:]
    val_labels = train_labels2[split_idx:]
    
    print(f"  ✓ Validación: {val_images.shape[0]} imágenes")

    # Combinar datasets de entrenamiento
    print("\nCombinando datasets de entrenamiento...")
    train_images = np.concatenate([train_images1, train_images2_train], axis=0)
    train_labels = np.concatenate([train_labels1, train_labels2_train], axis=0)

    # Verificar consistencia
    if train_images.shape[0] != train_labels.shape[0]:
        print(f"❌ ERROR: Desbalance")
        return None

    # Mezclar
    print("Mezclando aleatoriamente...")
    shuffle_indices = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_indices]
    train_labels = train_labels[shuffle_indices]

    print(f"\n📊 RESUMEN:")
    print(f"   Dataset combinado: {train_images.shape}")
    print(f"   Validación: {val_images.shape}")

    # Aplicar augmentación LIGERA
    print("\nAplicando augmentación de datos...")
    train_images = augment_images(train_images)

    # Crear modelo
    model = create_optimized_model((32, 32, 1), 62, use_dropout=True, l2_reg=0.001)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n🚀 INICIANDO ENTRENAMIENTO...")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping]
    )

    model.save("ocr_model.h5")
    print("\n✅ Modelo guardado")

    # Evaluar
    val_loss, val_accuracy = model.evaluate(val_images, val_labels)
    print(f"\n📊 Precisión en validación: {val_accuracy*100:.2f}%")

    return history
