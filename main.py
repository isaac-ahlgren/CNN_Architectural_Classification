import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

from inception import Inception
import tensorflow as tf
import random

def get_architecture_dataset(directory, image_height, image_width, batch_size):
     seed = random.getrandbits(32)

     train = tf.keras.utils.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(image_height,
    image_width), shuffle=True, seed=seed, validation_split=0.2, subset='training')
 
     test = tf.keras.utils.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=batch_size, image_size=(image_height,
    image_width), shuffle=True, seed=seed, validation_split=0.2, subset='validation')

     train = train.map(lambda x, y: (tf.divide(x, 255), y))
     test = test.map(lambda x, y: (tf.divide(x, 255), y))

     return train, test

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu',
                               padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2),
        Inception(32, (16,16), (16,16), 32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3)),
        tf.keras.layers.Dropout(0.5),
        Inception(64, (32,32), (32,32), 64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3)),
        tf.keras.layers.Dropout(0.5),
        #Inception(64, (32,32), (32,32), 64),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.MaxPool2D(pool_size=(3,3)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=7, activation='relu'),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(25)])

if __name__ == "__main__":
    # Parameters
    directory = "./architectural-styles-dataset"
    image_height = 256
    image_width = 256
    batch_size = 32
    epochs = 50
    learning_rate = 2e-4

    train, test = get_architecture_dataset(directory, image_height, image_width, batch_size)

    model = net()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(train,batch_size=batch_size,epochs=epochs)

    model.evaluate(test,batch_size=batch_size)

    tf.keras.utils.plot_model(model, show_shapes=True)


