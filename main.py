# =========================
# 1. IMPORT LIBRARIES
# =========================
# All libraries that were used some may not be used as they were there for testing new features that did not work
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import torch
from tensorflow.keras.callbacks import ReduceLROnPlateau
from google.colab import files
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from google.colab import drive
import kagglehub

# =========================
# 2. SET PATHS & GET LABELS
# =========================


directory = kagglehub.dataset_download("sperez07/apple-image-recognition-data") + '/sf_apple' # Downloads all the files

# Finds all the folders and assigns them a class
labels = os.listdir(directory)  
labels = [lbl for lbl in labels if os.path.isdir(os.path.join(directory, lbl))]
labels.sort()

# Number of classes to use for the training
num_fruit_to_classify = 18 
labels = labels[:num_fruit_to_classify]

# Lists the number of classes
num_classes = len(labels)

# Prints the name of all the classes
print(f"Found {num_classes} classes:")
print(labels)

# =========================
# 3. LOAD IMAGES & LABELS
# =========================
"""
Reads all images from subfolders (named by class), resizes them,
normalizes pixel values, and returns (X, y).
"""
def load_images_and_labels(root_dir, class_names):
    data = []
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(root_dir, class_name)

        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            try:
                # Resize the image
                img = load_img(img_path, target_size=(128, 128))

                # Normalize the pixel values to be between 0 and 1
                img_array = img_to_array(img) / 255
                data.append((img_array, idx))
            except:
                # Skip corrupted or unreadable files
                pass

    random.shuffle(data)
    X, y = zip(*data)
    return np.array(X), np.array(y)

X, y = load_images_and_labels(directory, labels)
print(f"Total images loaded: {len(X)}")
print("X shape:", X.shape, "| y shape:", y.shape)

# =========================
# 4. TRAIN/TEST SPLIT
# =========================

"""
Splits the data to what can be used to train the model 
and what be used to test it.
If we used them all to train then when we give it a random image that 
it used to train it would already know the answser.
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train set size:", len(X_train))
print("Test set size:", len(X_test))

# =========================
# 5. DATA AUGMENTATION
# =========================
# Alters the data so that there are slight changes to create more data
train_datagen = ImageDataGenerator(
    rotation_range= 0.4,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
# Activates the function
test_datagen = ImageDataGenerator()

# Calculate data statistics
train_datagen.fit(X_train)
test_datagen.fit(X_test)

# Convert y to one-hot encoding (changes labels of the class to numbers the CNN can use)
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_oh = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# =========================
# 6. BUILD A SIMPLE CNN MODEL
# =========================
"""
In this section, we will build our convolutional neural network (CNN) model
using Keras. We'll use a Sequential model that stacks layers in order.
"""
"""
1. We choose the model type, Sequential
2. We add Convolutional layers + Batch Normalization + Pooling layers.
3. We Flatten the result, then add Dense layers.
4. We finish with an output layer of size 'num_classes' (softmax for multi-class).
5. Finally, we compile the model (choose optimizer, loss function, metrics).
"""
# HINTS:
#  - 'filters' is how many feature detectors to learn in a Conv layer (e.g., 32, 64, 128).
#  - 'kernel_size' is the size of the sliding window (e.g., (3,3)).
#  - 'activation' is often 'relu' for convolution layers, 'softmax' for the last layer.
#  - 'input_shape' only needs to be specified in the first layer (height, width, channels).
#  - After Convolution, BatchNormalization can help stabilize and speed up training.
#  - MaxPooling2D((2,2)) halves the spatial dimensions of the feature maps.
#  - The final Dense layer must have 'num_classes' outputs for multi-class classification.

model = Sequential([
    # --- First Convolutional Block ---
    
    # Convolutional layer
    Conv2D(32 , (3 , 3), activation='relu', input_shape=(128 , 128 , 3)),

    # Batch Normalization helps stabilize training
    BatchNormalization(),

    # Pooling layer to downsample
    MaxPooling2D((2,2)),
    Dropout(0.1),

    # --- Second Convolutional Block ---
    Conv2D(64 , (3 , 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # --- Third Convolutional Block ---
    Conv2D(128 , (3 , 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),


    # --- Flatten & Dense Layers ---
    # Flatten the 3D feature maps into a 1D vector
    Flatten(),

    # Fully-connected layer (Dense).
    # 128 'Neurons' and activate ReLU
    Dense(128 , activation='relu'), 
    BatchNormalization(),

    # Final output layer: must match the number of classes (num_classes).
    # Use 'softmax' for multi-class classification.
    Dense(num_classes , activation='softmax')
])

# Show the model architecture
model.summary()

# =========================
# 7. TRAIN THE MODEL
# =========================

# Number of images the CNN will look at once
batch_size = 32 

# Changes the learning rate by monitoring val_accuracy
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)

# defines the optimizer, loss function, and metrics to use during training and evaluation
model.compile(
    optimizer=Adam(learning_rate=1e-1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors the val_loss to make sure there is progress
    patience=10,           # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore best weights from the epoch with the lowest validation loss
)

# Train the model with the learning rate scheduler and early stopping
history = model.fit(
    train_datagen.flow(X_train, y_train_oh, batch_size=batch_size),
    validation_data=test_datagen.flow(X_test, y_test_oh, batch_size=batch_size),
    epochs=100,  # Highest number of epochs the CNN can trian for but would most likely stop before that
    callbacks=[early_stopping, lr_scheduler]  # Add both early stopping and learning rate scheduler
)

# =========================
# 8. EVALUATE THE MODEL
# =========================

# Tests the model to see how well it does
print("\nEvaluating on test set:")
test_loss, test_acc = model.evaluate(test_datagen.flow(X_test, y_test_oh, batch_size=batch_size))
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# =========================
# 9. SAVE MODEL
# =========================
# Saves as a single Keras file
model.save('/content/drive/My Drive/model.keras') 
print("Model saved as model.keras")

# ===========================================
# 10. USING OUR OWN IMAGES TO TEST THE AI
# ===========================================

final_result = 0

# Mount the drive
drive.mount('/content/drive')

# Load the model (assuming it's saved as 'model.keras')
model = tf.keras.models.load_model('/content/drive/My Drive/model.keras')

# Path to your image
img_folder = '/content/drive/My Drive/apple'

#All categories
class_labels = ['sf_NIR_color_fresh_apple', 'sf_NIR_color_slightlyspoiled_apple', 'sf_NIR_color_spoiled_apple', 'sf_NIR_fresh_apple', 'sf_NIR_slightlyspoiled_apple', 'sf_NIR_spoiled_apple', 'sf_blue_color_fresh_apple', 'sf_blue_color_slightlyspoiled_apple', 'sf_blue_color_spoiled_apple', 'sf_blue_fresh_apple', 'sf_blue_slightlyspoiled_apple', 'sf_blue_spoiled_apple', 'sf_orange_color_fresh_apple', 'sf_orange_color_slightlyspoiled_apple', 'sf_orange_color_spoiled_apple', 'sf_orange_fresh_apple', 'sf_orange_slightlyspoiled_apple', 'sf_orange_spoiled_apple']


image_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

for img_path in image_files:
  # Load the image and resize it to the expected input size of the model (128x128)
  img = image.load_img(img_path, target_size=(128, 128))

  # Convert the image to a numpy array
  img_array = image.img_to_array(img)

  # Normalize the pixel values to be between 0 and 1
  img_array /= 255.0

  # Add a batch dimension since the model expects a batch of images
  img_array = np.expand_dims(img_array, axis=0)

  # Make a prediction
  prediction = model.predict(img_array)

  # Output the prediction (if it's a classification model)
  print("Prediction:", prediction)

  # Get the index of the highest predicted probability
  predicted_class_index = np.argmax(prediction)

  # Get the predicted class label
  predicted_class_label = class_labels[predicted_class_index]

  # Gives a values wether the the image indicates spoialge or not
  print(predicted_class_index)
  if predicted_class_index in [0, 3, 6, 9, 12, 15]:
    pass
    print("ns")
  elif predicted_class_index in [1, 4, 7, 10, 13, 16]:
    final_result += 1
    print("ss")
  elif predicted_class_index in [2, 5, 8, 11, 15, 17]:
    final_result += 2
    print("s")

  print(f"The model predicts the image is a {predicted_class_label}")

# Decides wether the image is fresh, slightly spoiled or fully spoiled
file_count = len([f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
print(file_count)
print(final_result)
if final_result == 0:
  print("The fruit is completely safe to eat")
elif final_result <= file_count:
  print("The fruit has some slight spoilage")
elif final_result > file_count:
  print("The fruit is spoiled")

# =========================
# 11. CLASSIFICATION REPORT
# =========================

# Gives a report on how the CNN did
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# =========================
# 12. VISUALIZE SOME RESULTS
# =========================
y_pred = np.argmax(model.predict(X_test), axis=-1)
def plot_samples(images, true_labels, pred_labels=None, n=12):
    """
    Plots a grid of sample images with their true and optionally predicted labels.
    """
    plt.figure(figsize=(12, 12))
    for i in range(n):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i])
        title_text = f"True: {labels[true_labels[i]]}"
        if pred_labels is not None:
            title_text += f"\nPred: {labels[pred_labels[i]]}"
        plt.title(title_text, fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show some test images with predictions
plot_samples(X_test, y_test, y_pred, n=12)

# =========================
# 13. TRAINING GRAPHS
# =========================
plt.figure(figsize=(12, 5))

# Plots graphs to see the rate of change of the CNN
# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

