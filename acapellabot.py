"""
Acapella extraction with a CNN

Typical usage:
    python acapellabot.py song.wav
    => Extracts acapella from <song.wav> to <song (Acapella Attempt).wav> using default weights

    python acapellabot.py --data input_folder --batch 32 --weights new_model_iteration.h5
    => Trains a new model based on song/acapella pairs in the folder <input_folder>
       and saves weights to <new_model_iteration.h5> once complete.
       See data.py for data specifications.
"""

import argparse
import random, string
import os

import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.models import Model

import console
import conversion
from data import Data


class AcapellaBot:
    def __init__(self):
        # Create model
        mashup = Input(shape=(None, None, 1), name='input')
        conv = Conv2D(64, 3, activation='relu', padding='same')(mashup)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = MaxPooling2D((2, 2), padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = MaxPooling2D((2, 2), padding='same')(conv)
        conv = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=1)(conv)
        conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
        conv = UpSampling2D((2, 2))(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = UpSampling2D((2, 2))(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
        conv = Conv2D(1, 3, activation='relu', padding='same')(conv)
        acapella = conv
        m = Model(inputs=mashup, outputs=acapella)
        console.log("Model has", m.count_params(), "params")
        m.compile(loss='mean_squared_error', optimizer='adam')
        self.model = m
        # need to know so that we can avoid rounding errors with spectrogram
        # this should represent how much the input gets downscaled
        # in the middle of the network
        self.peakDownscaleFactor = 4

    def train(self, data, epochs, batch=8):
        xTrain, yTrain = data.train()
        xValid, yValid = data.valid()
        while epochs > 0:
            console.log("Training for", epochs, "epochs on", len(xTrain), "examples")
            self.model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))
            console.notify(str(epochs) + " Epochs Complete!", "Training on", data.inPath, "with size", batch)
            while True:
                try:
                    epochs = int(input("How many more epochs should we train for? "))
                    break
                except ValueError:
                    console.warn("Oops, number parse failed. Try again, I guess?")
            if epochs > 0:
                save = input("Should we save intermediate weights [y/n]? ")
                if not save.lower().startswith("n"):
                    weightPath = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
                    console.log("Saving intermediate weights to", weightPath)
                    self.saveWeights(weightPath)


    def saveWeights(self, path):
        self.model.save_weights(path, overwrite=True)
    def loadWeights(self, path):
        self.model.load_weights(path)
    def isolateVocals(self, path, fftWindowSize, phaseIterations=10):
        console.log("Attempting to isolate vocals from", path)
        audio, sampleRate = conversion.loadAudioFile(path)
        spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize=fftWindowSize)
        console.log("Retrieved spectrogram; processing...")

        # newSpectrogram = self.model.predict(conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)[np.newaxis, :, :, np.newaxis])[0][:spectrogram.shape[0], :spectrogram.shape[1]]
        expandedSpectrogram = conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)
        expandedSpectrogramWithBatchAndChannels = expandedSpectrogram[np.newaxis, :, :, np.newaxis]
        predictedSpectrogramWithBatchAndChannels = self.model.predict(expandedSpectrogramWithBatchAndChannels)
        predictedSpectrogram = predictedSpectrogramWithBatchAndChannels[0, :, :, 0] # o /// o
        newSpectrogram = predictedSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]]
        console.log("Processed spectrogram; reconverting to audio")

        newAudio = conversion.spectrogramToAudioFile(newSpectrogram, fftWindowSize=fftWindowSize, phaseIterations=phaseIterations)
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])
        outputFileNameBase = os.path.join(pathParts[0], fileNameParts[0] + " (Acapella Attempt)")
        console.log("Converted to audio; writing to", outputFileNameBase)

        conversion.saveAudioFile(newAudio, outputFileNameBase + ".wav", sampleRate)
        conversion.saveSpectrogram(newSpectrogram, outputFileNameBase + ".png")
        conversion.saveSpectrogram(spectrogram, os.path.join(pathParts[0], fileNameParts[0] + " (Original)") + ".png")
        console.log("Vocal isolation complete 👌")

if __name__ == "__main__":
    # if data folder is specified, create a new data object and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="weights.h5", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int, help="Phase iterations for reconstruction")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    acapellabot = AcapellaBot()

    if len(args.files) == 0 and args.data:
        console.log("No files provided; attempting to train on " + args.data + "...")
        if args.load:
            console.h1("Loading Weights")
            acapellabot.loadWeights(args.weights)
        console.h1("Loading Data")
        data = Data(args.data, args.fft, args.split)
        console.h1("Training Model")
        acapellabot.train(data, args.epochs, args.batch)
        acapellabot.saveWeights(args.weights)
    elif len(args.files) > 0:
        console.log("Weights provided; performing inference on " + str(args.files) + "...")
        console.h1("Loading weights")
        acapellabot.loadWeights(args.weights)
        for f in args.files:
            acapellabot.isolateVocals(f, args.fft, args.phase)
    else:
        console.error("Please provide data to train on (--data) or files to infer on")
