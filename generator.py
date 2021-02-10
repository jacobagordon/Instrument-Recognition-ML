import math
import keras
from tensorflow.keras.utils import Sequence
import numpy as np
import dftstream as dft
import audioframes as af
import plottools as plot

import matplotlib.pyplot as plt
from scipy.io import wavfile

class DataGenerator(Sequence):

    def __init__(self, files, values, batch_size=100):
        self.batch_size = batch_size
        self.file_names = files
        self.values = values
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.file_names) / self.batch_size)

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_files = [self.file_names[i] for i in indexes]
        values = [self.values[i] for i in indexes]

        examples = self.__data__generation(batch_files)

        return examples, keras.utils.to_categorical(values)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.file_names))

    def __data__generation(self, batch_files):
        spectograms = list()
        minLength = 297

        for file in batch_files:
            audioframe = af.AudioFrames(file, 10, 30)
            dftframe = dft.DFTStream(audioframe)
            dftStack = np.vstack([item for item in dftframe])
            dftStack = dftStack.transpose()
            currentShape = np.shape(dftStack)

            if currentShape[1] < minLength:
                minLength = currentShape[1]
            spectograms.append(dftStack)

        for count, spect in enumerate(spectograms):
            spectograms[count] = spect[:200, :minLength]
            spectograms[count] = spectograms[count].flatten()

        examples = np.vstack([item for item in spectograms])

        return examples
