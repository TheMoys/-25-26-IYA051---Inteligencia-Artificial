import tensorflow as tf
from tensorflow.keras import layers, models

def create_optimized_model(input_shape, num_classes, use_dropout=True, l2_reg=0.0):
    """
    Modelo simplificado pero efectivo para OCR de 62 clases.
    """
    model = models.Sequential([
        # Entrada
        layers.Input(shape=input_shape),
        
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25) if use_dropout else layers.Layer(),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25) if use_dropout else layers.Layer(),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25) if use_dropout else layers.Layer(),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5) if use_dropout else layers.Layer(),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.Dropout(0.3) if use_dropout else layers.Layer(),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_simple_ocr_model(input_shape, num_classes):
    """
    Alias para compatibilidad.
    """
    return create_optimized_model(input_shape, num_classes, use_dropout=True, l2_reg=0.001)