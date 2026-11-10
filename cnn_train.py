import os
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import ImageFile
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------- Parameters --------------------
dataset_path = './medicinal_plants_dataset/'  # Root folder containing class subfolders
img_size = (224, 224)
batch_size = 32
num_epochs = 100

# -------------------- Load dataset --------------------
classes = [cls for cls in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, cls))]
num_classes = len(classes)
print(f"✅ Classes found: {classes}")

file_paths, labels = [], []

for idx, cls in enumerate(classes):
    cls_folder = os.path.join(dataset_path, cls)
    imgs = [os.path.join(cls_folder, img) for img in os.listdir(cls_folder)]
    file_paths.extend(imgs)
    labels.extend([idx] * len(imgs))

file_paths = np.array(file_paths)
labels = np.array(labels)

# One-hot encode labels
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, num_classes=num_classes)

# Shuffle dataset
file_paths, labels = shuffle(file_paths, labels, random_state=42)

# -------------------- Image preprocessing --------------------
def path_to_tensor(img_path, width=224, height=224):
    img = load_img(img_path, target_size=(width, height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x / 255.0  # Normalize

def paths_to_tensor(img_paths):
    tensors = [path_to_tensor(p) for p in tqdm(img_paths)]
    return np.vstack(tensors)

# Preprocess all images
X = paths_to_tensor(file_paths).astype('float32')
y = labels

# -------------------- Train/Test Split --------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------- CNN Model --------------------
def build_cnn_model(input_shape=(224,224,3), num_classes=num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(learning_rate=0.0004),
                  metrics=['accuracy'])
    return model

model = build_cnn_model()

# -------------------- Callbacks --------------------
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_medicinal_plant_cnn.h5', monitor='val_accuracy', save_best_only=True)

# -------------------- Train CNN --------------------
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=num_epochs,
                    batch_size=batch_size,
                    callbacks=[early_stop, checkpoint],
                    verbose=1)

# -------------------- Evaluate --------------------
loss, acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n✅ Final Validation Accuracy: {acc*100:.2f}%")

# -------------------- Plot Accuracy and Loss --------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.grid()
plt.show()
