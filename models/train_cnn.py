'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'C:/Users/Omkar/OneDrive/Documents/Cat_Dog_Classification/dataset/train', target_size=(64, 64), batch_size=32, class_mode='binary')

test_data = test_datagen.flow_from_directory(
    'C:/Users/Omkar/OneDrive/Documents/Cat_Dog_Classification/dataset/test', target_size=(64, 64), batch_size=32, class_mode='binary')

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_data, steps_per_epoch=8000 // 32, epochs=5, validation_data=test_data, validation_steps=2000 // 32)

# Save the Model
model.save('C:/Users/Omkar/OneDrive/Documents/Cat_Dog_Classification/cat_dog_model.h5')'''


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    '../dataset/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    '../dataset/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Build CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the Model
model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=25,
    validation_data=test_data,
    validation_steps=len(test_data),
    callbacks=[early_stopping, lr_reduction]
)

# Save the Model
model.save('../cat_dog_model.h5')
#model.save('../cat_dog_model.keras')
#model.save('C:/Users/Omkar/OneDrive/Documents/Cat_Dog_Classification/cat_dog_model.h5')
print("Model training completed and saved as '../cat_dog_model.h5'")