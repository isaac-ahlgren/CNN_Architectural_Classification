import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

from inception import Inception
from resnetblock import ResnetBlock
from TransitionBlock import TransitionBlock
from DenseBlock import DenseBlock
from keras import initializers
import tensorflow as tf
import random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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

def get_dataset(directory, image_height, image_width, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    seed = random.getrandbits(32)

    train_generator = train_datagen.flow_from_directory(directory, target_size=(image_height, image_width), color_mode='rgb', class_mode='sparse', batch_size=batch_size, shuffle=True, seed=seed, subset="training")

    test_generator = train_datagen.flow_from_directory(directory, target_size=(image_height, image_width), color_mode='rgb', class_mode='sparse', batch_size=batch_size, shuffle=True, seed=seed, subset="validation")
    
    train_dataset = tf.data.Dataset.from_generator(train_generator)

    test_generator = tf.data.Dataset.from_generator(test_generator)

    return train_dataset, test_dataset

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.RandomFlip("horizontal",
                         input_shape=(256,
                                     256,
                                     3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(filters=64, kernel_size=7, activation='relu',kernel_regularizer='l2'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=3),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu',kernel_regularizer='l2', strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=96, kernel_size=3, activation='relu',kernel_regularizer='l2', strides=2),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', kernel_regularizer='l2', strides=2),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Conv2D(filters=1024, kernel_size=3, activation='relu',kernel_regularizer='l2', strides=2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(25)])

def net2():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ResnetBlock(64, 2, first_block=True),
        tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(25)])
    
def resnet():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal",
                         input_shape=(256,
                                     256,
                                     3)),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Dropout(0.1),
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same',kernel_regularizer='l1'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(25)])

def googlenet():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal",
                         input_shape=(256,
                                     256,
                                     3)),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom(0.25),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, activation='relu'),
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(25)
    ])

def block_1():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal",
                         input_shape=(600,
                                     600,
                                     3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net

def densenet():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(25))
    return net

if __name__ == "__main__":
    # Parameters
    directory = "./architectural-styles-dataset"
    image_height = 600
    image_width = 600
    batch_size = 8
    epochs = 40
    learning_rate = 2e-4

    train, test = get_architecture_dataset(directory, image_height, image_width, batch_size)

    model = net2()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    model.fit(train,batch_size=batch_size,epochs=epochs)

    model.evaluate(test,batch_size=batch_size)

    model.summary()



