"""
Loads and stores mashup data given a folder full of acapellas and instrumentals
Assumes that all audio clips (wav, mp3) in the folder
a) have their Camelot key as the first token in the filename
b) are in the same BPM
c) have "acapella" somewhere in the filename if they're an acapella, and are otherwise instrumental
d) all have identical arrangements
e) have the same sample rate
"""
import sys
import os
import numpy as np
import h5py

import console
import conversion

# Modify these functions if your data is in a different format
def keyOfFile(fileName):
    firstToken = int(fileName.split()[0])
    if 0 < firstToken <= NUMBER_OF_KEYS:
        return firstToken
    console.warn("File", fileName, "doesn't specify its key, ignoring..")
    return None

def fileIsAcapella(fileName):
    return "acapella" in fileName.lower()


NUMBER_OF_KEYS = 12 # number of keys to iterate over
SLICE_SIZE = 128    # size of spectrogram slices to use

# Slice up matrices into squares so the neural net gets a consistent size for training (doesn't matter for inference)
def chop(matrix, scale):
    slices = []
    for time in range(0, matrix.shape[1] // scale):
        for freq in range(0, matrix.shape[0] // scale):
            s = matrix[freq * scale : (freq + 1) * scale,
                       time * scale : (time + 1) * scale]
            slices.append(s)
    return slices

class Data:
    def __init__(self, inPath, fftWindowSize=1536, trainingSplit=0.9):
        self.inPath = inPath
        self.fftWindowSize = fftWindowSize
        self.trainingSplit = trainingSplit
        self.x = []
        self.y = []
        self.load()
    def train(self):
        return (self.x[:int(len(self.x) * self.trainingSplit)], self.y[:int(len(self.y) * self.trainingSplit)])
    def valid(self):
        return (self.x[int(len(self.x) * self.trainingSplit):], self.y[int(len(self.y) * self.trainingSplit):])
    def load(self, saveDataAsH5=False):
        h5Path = os.path.join(self.inPath, "data.h5")
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")
            self.x = h5f["x"][:]
            self.y = h5f["y"][:]
        else:
            acapellas = {}
            instrumentals = {}
            # Hash bins for each camelot key so we can merge
            # in the future, this should be a generator w/ yields in order to eat less memory
            for i in range(NUMBER_OF_KEYS):
                key = i + 1
                acapellas[key] = []
                instrumentals[key] = []
            for dirPath, dirNames, fileNames in os.walk(self.inPath):
                for fileName in filter(lambda f : (f.endswith(".mp3") or f.endswith(".wav")) and not f.startswith("."), fileNames):
                    key = keyOfFile(fileName)
                    if key:
                        targetPathMap = acapellas if fileIsAcapella(fileName) else instrumentals
                        tag = "[Acapella]" if fileIsAcapella(fileName) else "[Instrumental]"
                        audio, sampleRate = conversion.loadAudioFile(os.path.join(self.inPath, fileName))
                        spectrogram, phase = conversion.audioFileToSpectrogram(audio, self.fftWindowSize)
                        targetPathMap[key].append(spectrogram)
                        console.info(tag, "Created spectrogram for", fileName, "in key", key, "with shape", spectrogram.shape)
            # Merge mashups
            for k in range(NUMBER_OF_KEYS):
                acapellasInKey = acapellas[k + 1]
                instrumentalsInKey = instrumentals[k + 1]
                count = 0
                for acapella in acapellasInKey:
                    for instrumental in instrumentalsInKey:
                        # Pad if smaller
                        if (instrumental.shape[1] < acapella.shape[1]):
                            newInstrumental = np.zeros(acapella.shape)
                            newInstrumental[:instrumental.shape[0], :instrumental.shape[1]] = instrumental
                            instrumental = newInstrumental
                        elif (acapella.shape[1] < instrumental.shape[1]):
                            newAcapella = np.zeros(instrumental.shape)
                            newAcapella[:acapella.shape[0], :acapella.shape[1]] = acapella
                            acapella = newAcapella
                        # simulate a limiter/low mixing (loses info, but that's the point)
                        # I've tested this against making the same mashups in Logic and it's pretty close
                        mashup = np.maximum(acapella, instrumental)
                        # chop into slices so everything's the same size in a batch
                        dim = SLICE_SIZE
                        mashupSlices = chop(mashup, dim)
                        acapellaSlices = chop(acapella, dim)
                        count += 1
                        self.x.extend(mashupSlices)
                        self.y.extend(acapellaSlices)
                console.info("Created", count, "mashups for key", k, "with", len(self.x), "total slices so far")
            # Add a "channels" channel to please the network
            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]
            # Save to file if asked
            if saveDataAsH5:
                h5f = h5py.File(h5Path, "w")
                h5f.create_dataset("x", data=self.x)
                h5f.create_dataset("y", data=self.y)
                h5f.close()

if __name__ == "__main__":
    # Simple testing code to use while developing
    console.h1("Loading Data")
    d = Data(sys.argv[1], 1536)
    console.h1("Writing Sample Data")
    conversion.saveSpectrogram(d.x[0], "x_sample_0.png")
    conversion.saveSpectrogram(d.y[0], "y_sample_0.png")
    audio = conversion.spectrogramToAudioFile(d.x[0], 1536)
    conversion.saveAudioFile(audio, "x_sample.wav", 22050)
