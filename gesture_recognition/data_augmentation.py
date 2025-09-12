import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ----------------------
# Parameters
# ----------------------
img_rows, img_cols = 128, 128
batch_size = 16
epochs = 20
num_classes = 5

train_dir = "./handgestures/train"
val_dir = "./handgestures/test"

# ----------------------
# Data generators
# ----------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.5,1.5],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    color_mode="rgb",  # MobileNetV2 expects 3 channels
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_rows, img_cols),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical"
)

# ----------------------
# Load pretrained MobileNetV2
# ----------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))
for layer in base_model.layers:
    layer.trainable = False  # freeze pretrained weights

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------
# Train
# ----------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# ----------------------
# Save model
# ----------------------
model.save("my_gestures_mobilenet.keras")

# ----------------------
# Save class labels
# ----------------------
labels = {v: k for k, v in train_generator.class_indices.items()}
with open("class_labels.json", "w") as f:
    json.dump(labels, f)

print("✅ Model and class_labels.json saved successfully!")
