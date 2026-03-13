import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Configuration
# Since this script is now in /backend/, we look for 'chest_xray' in the parent directory
DATASET_DIR = "../dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# If the directory '../chest_xray' exists, use it instead (as requested in the prompt)
if os.path.exists("../chest_xray"):
    DATASET_DIR = "../chest_xray"
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    VAL_DIR = os.path.join(DATASET_DIR, "val")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def create_data_generators():
    # Data Augmentation and Normalization for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only Normalization for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def build_model():
    # Load MobileNetV2 as base model, freeze its layers
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False

    # Add custom dense layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def main():
    print("Setting up data generators...")
    train_generator, val_generator, test_generator = create_data_generators()

    print("Building model...")
    model = build_model()
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        "pneumonia_model.h5", 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max",
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping]
    )

    print("Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Calculate F1 Score
    if (test_precision + test_recall) > 0:
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        print(f"Test F1 Score: {f1_score:.4f}")
    else:
        print("Test F1 Score: 0.0000")

    print("\nGenerating classification report and confusion matrix...")
    test_generator.reset()
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    target_names = list(test_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("Model training complete. Best model saved as 'pneumonia_model.h5'")

if __name__ == "__main__":
    main()
