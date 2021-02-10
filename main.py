import os
import time

# Add-on modules
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, \
    TimeDistributed, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Conv2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from keras.metrics import categorical_accuracy
from keras.callbacks import TensorBoard
import keras

import numpy as np
import generator as gn
import confusion as con

import matplotlib.pyplot as plt

from corpus import Corpus

def main():

    training_data = 'D:/PythonProjects/FinalProject/TestingTraining'
    test_data ='D:/PythonProjects/FinalProject/IRMAS-TestingData-Part1/IRMAS-TestingData-Part1/Part1'
    validation_data = 'D:/PythonProjects/FinalProject/TestingValidation'
    corpus = Corpus(training_data, test_data, validation_data, test_size=1500)

    train_gen = gn.DataGenerator(corpus.get_train_files(), corpus.get_train_labels())
    val_gen = gn.DataGenerator(corpus.get_validation_files(), corpus.get_validation_labels())
    test_gen = gn.DataGenerator(corpus.get_test_files(), corpus.get_test_labels())

    log_dir = os.path.join("logs", "{}".format(time.strftime('%d%b-%H%M')))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)

    model = Sequential()
    model.add(Dense(500, activation='relu'))
    model.add(Dense(350, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=[categorical_accuracy])

    history = model.fit(train_gen, epochs=40, callbacks=[tensorboard])

    model.summary()

    results = model.evaluate(val_gen)
    print("Validation Results: " + str(results[1]))
    predictions = model.predict(val_gen)
    predicted_values = np.argmax(predictions, axis=1)
    confusion, fig, ax, im = con.plot_confusion(predicted_values, corpus.get_validation_labels(), corpus.get_class_labels(), "Validation Matrix")

    results = model.evaluate(test_gen)
    predictions = model.predict_generator(test_gen)
    predicted_values = np.argmax(predictions, axis=1)
    confusion, fig, ax, im = con.plot_confusion(predicted_values, corpus.get_test_labels(), corpus.get_class_labels(), "Test Matrix")
    print("Test Results: " + str(results[1]))

if __name__ == '__main__':
    main()