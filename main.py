import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from skimage.color import rgb2lab, lab2rgb
from matplotlib import pyplot as plt
from Autoencoder import Autoencoder


def main():

    print("GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Load data
    (y_train, train_labels), (y_test, test_labels) = cifar10.load_data()

    # Choose only airplanes
    y_train = np.expand_dims(y_train, axis=1)[train_labels == 0]
    y_test = np.expand_dims(y_test, axis=1)[test_labels == 0]

    # Transform data and create datasets for neural network
    y_train = y_train.astype('float32') / 255.
    y_test = y_test.astype('float32') / 255.
    x_train = tf.expand_dims(rgb2lab(y_train)[:, :, :, 0], axis=-1)
    y_train = rgb2lab(y_train)[:, :, :, 1:] / 128.
    x_test = tf.expand_dims(rgb2lab(y_test)[:, :, :, 0], axis=-1)
    y_test = rgb2lab(y_test)[:, :, :, 1:] / 128.

    # Print shapes of datasets
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Create, compile and train model
    model = Autoencoder()
    model.compile(optimizer=Adam(),
                  loss='mean_squared_error',
                  metrics='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))

    # Test model on random image from test set
    test_index = np.random.randint(0, y_test.shape[0])
    X, y = x_test[test_index], y_test[test_index]
    y_pred = model.predict(np.expand_dims(X, axis=0))

    # Show result
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.imshow(tf.squeeze(X), cmap='gray')
    plt.title("Gray image")
    plt.axis('off')
    plt.subplot(132)
    img = np.zeros((32, 32, 3))
    img[:, :, 0] = tf.squeeze(X)
    img[:, :, 1:] = tf.squeeze(y_pred * 128)
    plt.imshow(lab2rgb(img))
    plt.title("Colorized image")
    plt.axis('off')
    plt.subplot(133)
    img[:, :, 1:] = tf.squeeze(y * 128)
    plt.imshow(lab2rgb(img))
    plt.title("Original image")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
