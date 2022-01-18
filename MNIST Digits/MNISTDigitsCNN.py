#This is a basic Convolutional Neural Network to solve the problem of classifying handwritten digits (which come from MNIST Database
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_output_classes = 10
img_rows, img_cols = 28, 28

def create_and_train_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(num_output_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2, validation_split = 0.1)
    return model


def load_model(model_name):
    return tf.keras.models.load_model(model_name)


def save_model(model_name):
    model.save(model_name)


def evaluate_model():
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("Test Loss: {}".format(val_loss))
    print("Test Accuracy: {}".format(val_acc))


def make_predictions():
    predictions = model.predict(x_test)
    num_of_tests = 500
    #Look at first 500 tests and show the wrong ones.
    for i in range(num_of_tests):
        if np.argmax(predictions[i]) != np.argmax(y_test[i]):
            print(np.argmax(predictions[i]))
            plt.imshow(x_test[i], cmap=plt.cm.binary)
            plt.show()


mnist = tf.keras.datasets.mnist

#preprocess the training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = tf.keras.utils.to_categorical(y_train, num_output_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_output_classes)

model = create_and_train_model()
#model = load_model("MNIST_Model.model")
evaluate_model()
model.save("MNIST_Model.model")
make_predictions()