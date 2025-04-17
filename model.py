import pandas as pd
import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 128

# Step 1: Load Excel
df = pd.read_excel("hb.xlsx")  # Ensure this Excel is in the same folder

# Step 2: Clean labels and remove quote marks from paths
df['classification'] = df['classification'].astype(str).str.strip().str.lower()
df['image ID'] = df['image ID'].astype(str).str.strip('"').str.strip()  # Strip surrounding quotes and whitespace

# Label mapping
label_map = {'moderate anemia': 0, 'mild anemia': 1, 'normal': 2}
df['label'] = df['classification'].map(label_map)

# Drop any rows with missing labels
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Step 3: Load and preprocess images
images = []
labels = []

image_folder = "Eyes_Dataset"  # Make sure this folder contains the images

for idx, row in df.iterrows():
    filename = row['image ID']
    img_path = os.path.join(image_folder, filename)

    print(f"[DEBUG] Checking file: {img_path}")

    if not os.path.exists(img_path):
        print(f"[WARNING] File not found: {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        images.append(img)
        labels.append(row['label'])
    else:
        print(f"[ERROR] Could not load image: {img_path}")

# Step 4: Prepare data for training
X = np.array(images)
y = np.array(labels)

if len(X) == 0:
    raise ValueError("[ERROR] No images loaded. Please check your Excel and image paths.")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define CNN model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

# Step 6: Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Step 7: Save the model
model.save("hemoglobin_model.h5")
print("[SUCCESS] Model saved as 'hemoglobin_model.h5'")





