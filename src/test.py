import numpy as np
import pandas as pd
from keras.utils import to_categorical

def prepare_data():
    """
    :return: train, valid, test sets
    """

    # X
    train_images = pd.read_csv('../data/train_images.csv').to_numpy()
    train_images = train_images.reshape((60000, 28, 28))
    train_images = train_images.astype('float32') / 255

    valid_images = train_images[50000:]
    train_images = train_images[:50000]

    test_images = pd.read_csv('../data/test_images.csv').to_numpy()
    test_images = test_images.reshape((10000, 28, 28))
    test_images = test_images.astype('float32') / 255

    # y
    train_labels = pd.read_csv('../data/train_labels.csv').to_numpy()
    train_labels = to_categorical(train_labels)

    valid_labels = train_labels[50000:]
    train_labels = train_labels[:50000]

    test_labels = pd.read_csv('../data/test_labels.csv').to_numpy()
    test_labels = to_categorical(test_labels)


    # MNIST 사이즈 2배로 늘리기 (28 -> 56)
    train_images = np.repeat(train_images, 2, axis=1)
    test_images = np.repeat(test_images, 2, axis=1)
    valid_images = np.repeat(valid_images, 2, axis=1)

    train_images = np.repeat(train_images, 2, axis=2)
    test_images = np.repeat(test_images, 2, axis=2)
    valid_images = np.repeat(valid_images, 2, axis=2)

    # GrayScale 을  RGB 로 바꾸기
    train_images = np.stack((train_images,) * 3, axis=-1)
    valid_images = np.stack((valid_images,) * 3, axis=-1)
    test_images = np.stack((test_images,) * 3, axis=-1)

    print('{:13} {}'.format('train_size', train_images.shape))
    print('{:13} {}'.format('test_size', test_images.shape))

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels


from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, BatchNormalization, Activation, Add, ZeroPadding2D
from keras.models import Model


def initialize_model():
    """
    :return: VGG 16 model
    """

    _input = Input((56, 56, 3))

    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv2)

    shortcut = pool1         # Skip Connection
    shortcut = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(shortcut)
    shortcut = MaxPooling2D((16, 16))(shortcut)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(pool1)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv3)
    pool2 = MaxPooling2D((2, 2))(conv4)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(pool2)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv5)
    conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv6)
    pool3 = MaxPooling2D((2, 2))(conv7)

    conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool3)
    conv9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv8)
    conv10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv9)
    pool4 = MaxPooling2D((2, 2))(conv10)

    conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(pool4)
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv11)
    conv13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv12)
    pool5 = MaxPooling2D((2, 2))(conv13)

    add = Add()([shortcut, pool5])  # Skip Connection

    flat = Flatten()(add)
    dense1 = Dense(4096, activation="relu")(flat)
    dense2 = Dense(1024, activation="relu")(dense1)
    output = Dense(10, activation="softmax")(dense2)

    model = Model(inputs=_input, outputs=output)

    print(model.summary())

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def infer_model(model, test_images, test_labels):
    """
    Test 데이터에 대하여 Accuracy 출력

    :param model: VGG16 model
    :param test_images:
    :param test_labels:

    :return:
    """
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('test_acc:', test_acc)


import matplotlib.pyplot as plt

def plot_history(history):

    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
    except:
        acc = history.history['acc']
        val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    f, ax = plt.subplots(1,2 , figsize=(10,5))

    ax[0].plot(epochs, acc, 'bo', label='Training acc')
    ax[0].plot(epochs, val_acc, 'b', label='Validation acc')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()

    ax[1].plot(epochs, loss, 'ro', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()




if __name__ == '__main__':

    epochs = 5
    batch_size = 64
    train_images, train_labels,  valid_images, valid_labels, test_images, test_labels = prepare_data()
    
    model = initialize_model()
    history = model.fit(x=train_images, y=train_labels,
                        validation_data=(valid_images, valid_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2)

    plot_history(history)
    infer_model(model, test_images, test_labels)
