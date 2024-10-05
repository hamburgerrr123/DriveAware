# model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print(f"TensorFlow version: {tf.__version__}")
print("Current working directory:", os.getcwd())
print("Contents of current directory:", os.listdir())

if os.path.exists('data'):
    print("Contents of 'data' directory:", os.listdir('data'))
    print("Contents of 'data/train' directory:", os.listdir('data/train'))
    print("Contents of 'data/test' directory:", os.listdir('data/test'))
else:
    print("'data' directory not found")

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

BS = 32
TS = (24,24)
train_dir = 'data/train'
valid_dir = 'data/test'

train_batch = generator(train_dir, shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator(valid_dir, shuffle=True, batch_size=BS, target_size=TS)

SPE = len(train_batch.classes) // BS
VS = len(valid_batch.classes) // BS
print(f"Steps per epoch: {SPE}, Validation steps: {VS}")

model = Sequential([
    Input(shape=(24, 24, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # 4 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_accuracy')

history = model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=15,
    steps_per_epoch=SPE,
    validation_steps=VS,
    callbacks=[early_stopping, model_checkpoint]
)

# Save final model
model.save('models/cnnCat2.keras')
print("Model saved successfully.")

# Optional: Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()