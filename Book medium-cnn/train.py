import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator

TRAIN_FOLDER = './dogscats/train/'
VAL_FOLDER = './dogscats/valid/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    rescale=1./255
)
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_FOLDER,
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=1
)
print('Labels:', dict([(i, k) for i, k in enumerate(train_generator.class_indices)]))

val_generator = val_datagen.flow_from_directory(
    directory=VAL_FOLDER,
    target_size=(32, 32),
    color_mode="rgb",
    batch_size=32,
    class_mode="binary",
    shuffle=True,
    seed=1
)

# Cargamos la clase para generar modelos sequenciales
from keras.models import Sequential 
# Cargamos las siguientes capas
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
# Bloque 1:
# - Conv1a: neuronas=128, ventana=(3,3), activacion=ReLU.
# - Conv1b: neuronas=256, ventana=(3,3), activacion=ReLU.
# - Max-pooling: ventana=(3,3), stride=2.
model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(256, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(3, strides=(2, 2)))

# Bloque 2:
# - Conv2a: neuronas=128, ventana=(3,3), activacion=ReLU.
# - Conv2b: neuronas=64, ventana=(3,3), activacion=ReLU.
# - Max-pooling: ventana=(3,3), stride=2.
model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(3, strides=(2, 2)))

# Convertimos lo que quede (HxWxC) a un vector-columna
model.add(Flatten())

# Bloque Fully-Connected
# - FC1: neuronas=64, activacion=ReLU
# - FC2: neuronas=1, activacion=Sigmoid
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=val_generator,
        validation_steps=800)

model_json = model.to_json()
with open('model.json', 'w') as json_file:
	json_file.write(model_json)

model.save_weights('model.h5')
print("Guardado con exito.")