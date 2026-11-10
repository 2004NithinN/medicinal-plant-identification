import os
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_files
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------- Load Dataset --------------------
dataset_path = './medicinal_plants_dataset/'

data = load_files(dataset_path)
files = np.array(data['filenames'])
labels = np.array(data['target'])
num_classes = len(np.unique(labels))
print(f"✅ Detected {num_classes} plant classes.")

targets = to_categorical(labels, num_classes=num_classes)

# -------------------- Preprocessing --------------------
def path_to_tensor(img_path, width=224, height=224):
    img = load_img(img_path, target_size=(width, height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)  # MobileNetV2 preprocessing

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

print("🔄 Preprocessing images...")
tensors = paths_to_tensor(files).astype('float32')

# -------------------- Train/Test Split --------------------
X_train, X_val, y_train, y_val = train_test_split(
    tensors, targets, test_size=0.2, random_state=42, stratify=targets
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# -------------------- Data Augmentation --------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# -------------------- MobileNetV2 Base Model (Local Weights) --------------------
# Place the weights file in your project folder, e.g., 'mobilenet_v2_weights.h5'
weights_path = './mobilenet_v2_weights.h5'  

base_model = MobileNetV2(
    weights=None,       
    include_top=False,
    input_shape=(224, 224, 3)
)

# Load local weights
if os.path.exists(weights_path):
    base_model.load_weights(weights_path)
    print("✅ MobileNetV2 weights loaded from local file.")
else:
    raise FileNotFoundError(f"MobileNetV2 weights not found at {weights_path}")

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# -------------------- Add Custom Layers --------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------- Callbacks --------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_mobilenetv2_model.h5', monitor='val_accuracy', save_best_only=True)

# -------------------- Train Model --------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -------------------- Evaluate Model --------------------
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")

# -------------------- Plot Accuracy and Loss --------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.grid()
plt.show()
