import os.path

import numpy as np
import random
import scipy.io.wavfile
import keras


class Corpus:

    instruments = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

    instruments2int = dict([(key, val) for val, key in enumerate(instruments)])

    extension = '.wav'

    def __init__(self, train_file_dir, test_file_dir, validation_file_dir, size=0, test_size=0, validation_size=0):

        self.files = os.path.realpath(train_file_dir)
        self.size = size
        self.dataset_len = len(train_file_dir)

        self.wav_files = []
        self.instrument_labels = []
        self.instrument_labels_num = []

        self.validation_files = os.path.realpath(validation_file_dir)
        self.validation_wav_files = []
        self.validation_instrument_labels = []
        self.validation_instrument_labels_num = []
        self.validation_size = validation_size

        self.test_files = os.path.realpath(test_file_dir)
        self.test_wav_files = []
        self.test_instrument_labels = []
        self.test_instrument_labels_num = []
        self.test_size = test_size

        self.class_labels = ("cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi")

        self.load_train()
        self.load_test()
        self.load_validation()

    def load_train(self):
        # Get train files
        for root, dirs, files in os.walk(self.files):
            for d in dirs:
                for path in os.listdir(os.path.join(root, d)):
                    full_path = os.path.join(root, d, path)
                    if os.path.isfile(full_path):
                        self.wav_files.append(full_path)
                        self.instrument_labels.append(d)

        self.instrument_labels_num = ([self.instruments2int[i] for i in self.instrument_labels])

        # Shuffle values
        shuffle = list(zip(self.wav_files, self.instrument_labels_num))
        random.shuffle(shuffle)
        self.wav_files, self.instrument_labels_num = zip(*shuffle)

        # Truncate if needed
        if self.size > 0:
            self.wav_files = self.wav_files[:self.size]
            self.instrument_labels_num = self.instrument_labels_num[:self.size]

    def load_test(self):
        # Get test files
        for root, dirs, files in os.walk(self.test_files):
            for f in files:
                fileName, file_extension = os.path.splitext(f)
                file_extension.lower()
                if file_extension == self.extension:
                    self.test_wav_files.append(os.path.join(root, f))
                    fileName = fileName + '.txt'
                    instru = open(os.path.join(root, fileName))
                    self.test_instrument_labels.append(instru.readline().strip())

        self.test_instrument_labels_num = ([self.instruments2int[i] for i in self.test_instrument_labels])

        # shuffle values
        shuffle = list(zip(self.test_wav_files, self.test_instrument_labels_num))
        random.shuffle(shuffle)
        self.test_wav_files, self.test_instrument_labels_num = zip(*shuffle)

        # Truncate if needed
        if self.test_size > 0:
            self.test_wav_files = self.test_wav_files[:self.test_size]
            self.test_instrument_labels_num = self.test_instrument_labels_num[:self.test_size]

    def load_validation(self):
        # Get validation files
        for root, dirs, files in os.walk(self.validation_files):
            for d in dirs:
                for path in os.listdir(os.path.join(root, d)):
                    full_path = os.path.join(root, d, path)
                    if os.path.isfile(full_path):
                        self.validation_wav_files.append(full_path)
                        self.validation_instrument_labels.append(d)

        self.validation_instrument_labels_num = ([self.instruments2int[i] for i in self.validation_instrument_labels])

        # shuffle values
        shuffle = list(zip(self.validation_wav_files, self.validation_instrument_labels_num))
        random.shuffle(shuffle)
        self.validation_wav_files, self.validation_instrument_labels_num = zip(*shuffle)

        # Truncate if needed
        if self.validation_size > 0:
            self.validation_wav_files = self.validation_wav_files[:self.validation_size]
            self.validation_instrument_labels_num = self.validation_instrument_labels_num[:self.validation_size]

    def get_train_files(self):
        return self.wav_files

    def get_train_labels(self):
        return self.instrument_labels_num

    def get_validation_files(self):
        return self.validation_wav_files

    def get_validation_labels(self):
        return self.validation_instrument_labels_num

    def get_test_files(self):
        return self.test_wav_files

    def get_test_labels(self):
        return self.test_instrument_labels_num

    def get_class_labels(self):
        return self.class_labels
