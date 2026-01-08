import numpy as np
import tensorflow as tf
from dataset import load_dataset
from model import create_optimized_model
import matplotlib.pyplot as plt
import pickle


def augment_images(images):
    """
    Aplica augmentación de datos básica utilizando TensorFlow.
    """
    augmented_images = []
    for img in images:
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        
        # Aplicar transformaciones aleatorias
        img_tensor = tf.image.random_flip_left_right(img_tensor)
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.1)
        img_tensor = tf.image.random_contrast(img_tensor, lower=0.9, upper=1.1)
        
        augmented_images.append(img_tensor.numpy())
    
    return np.array(augmented_images)

def train_model():
    """
    Entrena el modelo OCR con configuración optimizada.
    """
    print("=" * 80)
    print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO OCR")
    print("=" * 80)
    
    # ============================================================================
    # PASO 1: CARGAR DATASETS
    # ============================================================================
    print("\n📂 CARGANDO DATASETS...")
    
    # Dataset 1: Tipografías digitales
    print("\n1️⃣  Cargando DatasetCompleto2 (tipografías digitales)...")
    images1, labels1 = load_dataset("./dataset/DatasetCompleto2", dataset_type="new")
    print(f"    ✓ Cargadas {images1.shape[0]} imágenes")
    
    # Dataset 2: Manuscritas digitalizadas
    print("\n2️⃣  Cargando datasetCompleto (manuscritas digitalizadas)...")
    images2, labels2 = load_dataset("./dataset/datasetCompleto", dataset_type="old")
    print(f"    ✓ Cargadas {images2.shape[0]} imágenes")
    
    # ============================================================================
    # PASO 2: COMBINAR TODOS LOS DATASETS PRIMERO
    # ============================================================================
    print("\n🔀 COMBINANDO TODOS LOS DATASETS...")
    
    all_images = np.concatenate([images1, images2], axis=0)
    all_labels = np.concatenate([labels1, labels2], axis=0)
    
    print(f"    ✓ Total imágenes combinadas: {all_images.shape[0]}")
    
    # ============================================================================
    # PASO 3: MEZCLAR ALEATORIAMENTE TODO
    # ============================================================================
    print("\n🎲 MEZCLANDO TODOS LOS DATOS...")
    
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    # ============================================================================
    # PASO 4: DIVIDIR EN TRAIN/VALIDATION (80/20)
    # ============================================================================
    print("\n📊 DIVIDIENDO EN TRAIN/VALIDATION (80% / 20%)...")
    
    split_idx = int(len(all_images) * 0.8)
    
    train_images = all_images[:split_idx]
    train_labels = all_labels[:split_idx]
    val_images = all_images[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"    ✓ Training set: {train_images.shape[0]} imágenes")
    print(f"    ✓ Validation set: {val_images.shape[0]} imágenes")
    
    # ============================================================================
    # PASO 5: VERIFICAR DATOS
    # ============================================================================
    print("\n🔍 VERIFICACIÓN DE DATOS:")
    print(f"    Train images shape: {train_images.shape}")
    print(f"    Train labels shape: {train_labels.shape}")
    print(f"    Val images shape: {val_images.shape}")
    print(f"    Val labels shape: {val_labels.shape}")
    print(f"    Labels únicos en train: {len(np.unique(train_labels))}")
    print(f"    Labels únicos en val: {len(np.unique(val_labels))}")
    print(f"    Rango de labels: [{np.min(train_labels)}, {np.max(train_labels)}]")
    
    # Verificar distribución de clases
    print("\n📊 DISTRIBUCIÓN DE CLASES:")
    for label in range(62):
        count_train = np.sum(train_labels == label)
        count_val = np.sum(val_labels == label)
        
        if count_train == 0 or count_val == 0:
            print(f"    ⚠️  Clase {label}: train={count_train}, val={count_val}")
    
    missing_train = np.sum([np.sum(train_labels == i) == 0 for i in range(62)])
    missing_val = np.sum([np.sum(val_labels == i) == 0 for i in range(62)])
    
    if missing_train > 0 or missing_val > 0:
        print(f"    ⚠️  Clases faltantes - Train: {missing_train}, Val: {missing_val}")
    else:
        print(f"    ✅ Todas las 62 clases tienen ejemplos en ambos sets")
    
    # ============================================================================
    # PASO 6: NO APLICAR DATA AUGMENTATION (para debug)
    # ============================================================================
    print("\n⚠️  ENTRENANDO SIN DATA AUGMENTATION (para debug)")
    # train_images = augment_images(train_images)  # ← COMENTADO
    
    # ============================================================================
    # PASO 7: CREAR MODELO
    # ============================================================================
    print("\n🏗️  CREANDO MODELO...")
    
    model = create_optimized_model(
        input_shape=(32, 32, 1),
        num_classes=62,
        use_dropout=True,
        l2_reg=0.0001
    )
    
    print("\n📐 ARQUITECTURA DEL MODELO:")
    model.summary()
    
    # ============================================================================
    # PASO 8: CALLBACKS
    # ============================================================================
    print("\n⚙️  CONFIGURANDO CALLBACKS...")
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",  # ← Cambiar a accuracy
        patience=20,  # ← Más paciencia
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # ============================================================================
    # PASO 9: COMPILAR MODELO
    # ============================================================================
    print("\n⚙️  COMPILANDO MODELO...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),  # ← Learning rate un poco más alto
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("    ✓ Optimizer: Adam (lr=0.0003)")
    print("    ✓ Loss: sparse_categorical_crossentropy")
    print("    ✓ Metrics: accuracy")
    
    # ============================================================================
    # PASO 10: ENTRENAR
    # ============================================================================
    print("\n" + "=" * 80)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 80)
    print(f"    Epochs: 50")
    print(f"    Batch size: 32")
    print(f"    Learning rate inicial: 0.0003")
    print(f"    Early stopping patience: 20")
    print("=" * 80 + "\n")
    
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    # ============================================================================
    # PASO 11: GUARDAR HISTORIAL
    # ============================================================================
    print("\n💾 GUARDANDO HISTORIAL DE ENTRENAMIENTO...")
    
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("    ✓ Historial guardado en 'training_history.pkl'")
    
    # ============================================================================
    # PASO 12: GENERAR GRÁFICAS
    # ============================================================================
    print("\n📊 GENERANDO GRÁFICAS...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    axes[0].set_title('Loss durante entrenamiento', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='green')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='red')
    axes[1].set_title('Accuracy durante entrenamiento', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("    ✓ Gráficas guardadas en 'training_history.png'")
    plt.close()
    
    # ============================================================================
    # PASO 13: GUARDAR MODELO FINAL
    # ============================================================================
    print("\n💾 GUARDANDO MODELO FINAL...")
    
    model.save("ocr_model.h5")
    print("    ✓ Modelo guardado en 'ocr_model.h5'")
    
    # ============================================================================
    # PASO 14: EVALUACIÓN FINAL
    # ============================================================================
    print("\n" + "=" * 80)
    print("📊 EVALUACIÓN FINAL")
    print("=" * 80)
    
    train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=0)
    print(f"\n📈 TRAINING SET:")
    print(f"    Loss: {train_loss:.4f}")
    print(f"    Accuracy: {train_accuracy*100:.2f}%")
    
    val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\n📉 VALIDATION SET:")
    print(f"    Loss: {val_loss:.4f}")
    print(f"    Accuracy: {val_accuracy*100:.2f}%")
    
    overfitting = train_accuracy - val_accuracy
    print(f"\n🔍 OVERFITTING CHECK:")
    print(f"    Diferencia: {overfitting*100:.2f}%")
    
    if overfitting > 0.15:
        print(f"    ⚠️  ADVERTENCIA: Posible overfitting (diferencia > 15%)")
    else:
        print(f"    ✅ Generalización aceptable")
    
    # ============================================================================
    # PASO 15: ALERTAS
    # ============================================================================
    print("\n" + "=" * 80)
    
    if val_accuracy < 0.5:
        print("❌ ACCURACY MUY BAJA - POSIBLES CAUSAS:")
        print("   1. Learning rate muy alto/bajo")
        print("   2. Datos mal etiquetados")
        print("   3. Modelo inadecuado para 62 clases")
        print("   4. Necesita más epochs")
    elif val_accuracy < 0.7:
        print("⚠️  ACCURACY MODERADA - SUGERENCIAS:")
        print("   1. Aumentar epochs")
        print("   2. Ajustar learning rate")
        print("   3. Agregar más data augmentation")
    elif val_accuracy < 0.85:
        print("✅ ACCURACY BUENA - Modelo funcional")
    else:
        print("🎉 ACCURACY EXCELENTE - Modelo óptimo")
    
    print("=" * 80)
    
    # ============================================================================
    # RESUMEN FINAL
    # ============================================================================
    print("\n📋 RESUMEN:")
    print(f"    Total imágenes entrenamiento: {len(train_images)}")
    print(f"    Total imágenes validación: {len(val_images)}")
    print(f"    Epochs ejecutados: {len(history.history['loss'])}")
    print(f"    Mejor val_accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"    Archivos generados:")
    print(f"        - ocr_model.h5")
    print(f"        - best_model.h5")
    print(f"        - training_history.pkl")
    print(f"        - training_history.png")
    
    print("\n✅ ENTRENAMIENTO COMPLETADO\n")
    
    return history
