from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Directorio del dataset
dataset_dir = r"C:\Users\Aureli\Dropbox\PC (2)\Documents\Git_works_MASTER_AI\Master_IA\AppNudisTFM\nudibranquios_dataset\train"

# Generador de datos con augmentación y partición
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generador para entrenamiento
train_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',  # MULTICLASE
    subset='training'
)

# Generador para validación
val_gen = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
