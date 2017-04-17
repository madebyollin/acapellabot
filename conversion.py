import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
from os.path import basename
from math import ceil
import argparse
import console

def loadAudioFile(filePath):
    audio, sampleRate = librosa.load(filePath)
    return audio, sampleRate

def saveAudioFile(audioFile, filePath, sampleRate):
    librosa.output.write_wav(filePath, audioFile, sampleRate, norm=True)
    console.info("Wrote audio file to", filePath)

def expandToGrid(spectrogram, gridSize):
    # crop along both axes
    newY = ceil(spectrogram.shape[1] / gridSize) * gridSize
    newX = ceil(spectrogram.shape[0] / gridSize) * gridSize
    newSpectrogram = np.zeros((newX, newY))
    newSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
    return newSpectrogram

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioFile, fftWindowSize):
    spectrogram = librosa.stft(audioFile, fftWindowSize)
    phase = np.imag(spectrogram)
    amplitude = np.log1p(np.abs(spectrogram))
    return amplitude, phase

# This is the nutty one
def spectrogramToAudioFile(spectrogram, fftWindowSize, phaseIterations=10, phase=None):
    if phase is not None:
        # reconstructing the new complex matrix
        squaredAmplitudeAndSquaredPhase = np.power(spectrogram, 2)
        squaredPhase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase - squaredPhase, 0))
        amplitude = np.expm1(unexpd)
        stftMatrix = amplitude + phase * 1j
        audio = librosa.istft(stftMatrix)
    else:
        # phase reconstruction with successive approximation
        # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
        # for the algorithm used
        amplitude = np.exp(spectrogram) - 1
        for i in range(phaseIterations):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * (2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, fftWindowSize)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio

def loadSpectrogram(filePath):
    fileName = basename(filePath)
    if filePath.index("sampleRate") < 0:
        console.warn("Sample rate should be specified in file name", filePath)
        sampleRate == 22050
    else:
        sampleRate = int(fileName[fileName.index("sampleRate=") + 11:fileName.index(").png")])
    console.info("Using sample rate : " + str(sampleRate))
    image = io.imread(filePath, as_grey=True)
    return image / np.max(image), sampleRate

def saveSpectrogram(spectrogram, filePath):
    spectrum = spectrogram
    console.info("Range of spectrum is " + str(np.min(spectrum)) + " -> " + str(np.max(spectrum)))
    image = np.clip((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)), 0, 1)
    console.info("Shape of spectrum is", image.shape)
    # Low-contrast image warnings are not helpful, tyvm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(filePath, image)
    console.log("Saved image to", filePath)

def fileSuffix(title, **kwargs):
    return " (" + title + "".join(sorted([", " + i + "=" + str(kwargs[i]) for i in kwargs])) + ")"

def handleAudio(filePath, args):
    console.h1("Creating Spectrogram")
    INPUT_FILE = filePath
    INPUT_FILENAME = basename(INPUT_FILE)

    console.info("Attempting to read from " + INPUT_FILE)
    audio, sampleRate = loadAudioFile(INPUT_FILE)
    console.info("Max of audio file is " + str(np.max(audio)))
    spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize=args.fft)
    SPECTROGRAM_FILENAME = INPUT_FILENAME + fileSuffix("Input Spectrogram", fft=args.fft, iter=args.iter, sampleRate=sampleRate) + ".png"

    saveSpectrogram(spectrogram, SPECTROGRAM_FILENAME)

    print()
    console.wait("Saved Spectrogram; press Enter to continue...")
    print()

    handleImage(SPECTROGRAM_FILENAME, args, phase)


def handleImage(fileName, args, phase=None):
    console.h1("Reconstructing Audio from Spectrogram")

    spectrogram, sampleRate = loadSpectrogram(fileName)
    audio = spectrogramToAudioFile(spectrogram, fftWindowSize=args.fft, phaseIterations=args.iter)

    sanityCheck, phase = audioFileToSpectrogram(audio, fftWindowSize=args.fft)
    saveSpectrogram(sanityCheck, fileName + fileSuffix("Output Spectrogram", fft=args.fft, iter=args.iter, sampleRate=sampleRate) + ".png")

    saveAudioFile(audio, fileName + fileSuffix("Output", fft=args.fft, iter=args.iter) + ".wav", sampleRate)

if __name__ == "__main__":
    # Test code for experimenting with modifying acapellas in image processors (and generally testing the reconstruction pipeline)
    parser = argparse.ArgumentParser(description="Convert image files to audio and audio files to images")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--iter", default=10, type=int, help="Number of iterations to use for phase reconstruction")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    for f in args.files:
        if (f.endswith(".mp3") or f.endswith(".wav")):
            handleAudio(f, args)
        elif (f.endswith(".png")):
            handleImage(f, args)
