from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Convolution2D,Activation,Flatten,Dense,Dropout,MaxPool2D,BatchNormalization


class VGG:

    @staticmethod
    def build(width, height, depth, classes):

        input_shape = (height, width, depth)
        model = Sequential()

        model.add(Conv2D(8, kernel_size=(3, 3), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(3, 3)))

        model.add(Conv2D(16, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.summary()

        return model