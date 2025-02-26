import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image 
from tensorflow.keras.preprocessing import image_dataset_from_directory, image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = "Dataset"

def clean_dataset(folder_path):
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            
            if not any(file.lower().endswith(ext) for ext in allowed_extensions):
                print(f"🛑 Removing non-image file: {file_path}")
                os.remove(file_path)
                continue 
            
            try:
                img = Image.open(file_path)
                img.verify() 
                img.close()
                img = Image.open(file_path).convert("RGB")
                img.save(file_path)
                
            except (IOError, SyntaxError):
                print(f"⚠️ Corrupt image found and removed: {file_path}")
                os.remove(file_path)

print("🔍 Checking dataset for corrupt or invalid images...")
clean_dataset(dataset_path)
print("✅ Dataset cleaning completed!\n")

batch_size = 32
img_size = (255, 255)

train_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_dataset.class_names
print("Classes:", class_names)

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(255,255,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


epochs = 10
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

loss, acc = model.evaluate(val_dataset)
print(f"✅ Validation Accuracy: {acc * 100:.2f}%")

model.save('terrain.h5')

# def predict_image(image_path):
#     img = image.load_img(image_path, target_size=(255, 255))
#     img_array = image.img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     predictions = model.predict(img_array)
#     predicted_class = class_names[np.argmax(predictions)]
    
#     return predicted_class

# print("Predicted Terrain:", predict_image("image.png"))
