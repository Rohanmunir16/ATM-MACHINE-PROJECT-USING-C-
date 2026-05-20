import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

# ============================================================================
# 1. ENVIRONMENT CONFIGURATION (UPDATED WITH CORRECT NESTED PATH)
# ============================================================================
# Point this directly to the deepest nested folder where train, valid, and test actually live
DATASET_BASE_DIR = "/content/dataset/real_vs_fake/real-vs-fake"

TRAIN_DIR = os.path.join(DATASET_BASE_DIR, "train")
VAL_DIR = os.path.join(DATASET_BASE_DIR, "valid")
TEST_DIR = os.path.join(DATASET_BASE_DIR, "test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 7
LEARNING_RATE = 1e-4

# ============================================================================
# 2. SECURE NATIVE RGB PIPELINES (Bypasses manual BGR scaling traps)
# ============================================================================
print("Setting up training data pipeline...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

print("\nSetting up validation data pipeline...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

print("\nSetting up isolated testing data pipeline...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Enable ultra-fast background prefetching memory allocation
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================================
# 3. CONFLICT-FREE MODEL GENERATOR DEFINITIONS
# ============================================================================

def build_conflict_free_cnn():
    """
    Model 1: Custom Architecture with hardcoded ImageNet structural normalization
    built right into the input tensor logic.
    """
    inputs = layers.Input(shape=(224, 224, 3), name="raw_rgb_input")
    
    # Mathematical rescaling layer converts raw RGB [0-255] pixels to mean-centered intervals
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229**2, 0.224**2, 0.225**2])(x)
    
    # Clean feature layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Dense Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="spatial_output")(x)
    
    return models.Model(inputs, outputs, name="Production_Custom_CNN")


def build_stable_mobilenet_v2():
    """
    Model 2: Safe Transfer Learning Backbone using standard base architectures
    to isolate dependency loading breaks entirely.
    """
    inputs = layers.Input(shape=(224, 224, 3), name="raw_rgb_input")
    
    # Hardcoded Rescaling maps raw inputs directly into the [-1, 1] frame expected by MobileNetV2
    x = layers.Rescaling(1./127.5, offset=-1.0)(inputs)
    
    # Load vanilla architecture safely
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    
    # Freeze initial layers to protect initial weights and reduce overhead
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False
        
    x = base_model(x, training=True)
    
    # Production deployment classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid', name="mobilenet_output")(x)
    
    return models.Model(inputs, outputs, name="Production_MobileNetV2")


# ============================================================================
# 4. RUN SYSTEM TRAINING & BENCHMARKING
# ============================================================================

# --- Execution Block: Model 1 ---
print("\n" + "="*50 + "\nTRAINING ENGINE: Custom CNN\n" + "="*50)
cnn_model = build_conflict_free_cnn()
cnn_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("\nEvaluating Model 1 on Test Set...")
cnn_model.evaluate(test_ds)
cnn_model.save("final_production_cnn.keras")


# --- Execution Block: Model 2 ---
print("\n" + "="*50 + "\nTRAINING ENGINE: MobileNetV2\n" + "="*50)
mobilenet_model = build_stable_mobilenet_v2()
mobilenet_model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
mobilenet_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("\nEvaluating Model 2 on Test Set...")
mobilenet_model.evaluate(test_ds)
mobilenet_model.save("final_production_mobilenet.keras")

print("\n🎉 All models trained and saved safely as native .keras files!")