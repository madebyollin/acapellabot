import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
from os.path import basename
from math import ceil
import argparse
import console

# Loads an audio file and returns audio data, sample rate
def loadAudioFile(filePath):
    audio, sampleRate = librosa.load(filePath)
    return audio, sampleRate

def saveAudioFile(audioFile, filePath, sampleRate):
    librosa.output.write_wav(filePath, audioFile, sampleRate, norm=True)
    console.info("Wrote audio file to ", filePath)

def resizeToGrid(spectrogram, gridSize):
    # crop along both axes
    newY = (spectrogram.shape[1] // gridSize) * gridSize
    newX = (spectrogram.shape[0] // gridSize) * gridSize
    newSpectrogram = spectrogram[:newX, :newY]
    return newSpectrogram

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioFile, fftWindowSize=2048, gridSnap=1):
    spectrogram = librosa.stft(audioFile, fftWindowSize)
    phase = np.imag(spectrogram)
    amplitude = np.log1p(np.abs(spectrogram))
    if (gridSnap > 1):
        amplitude = resizeToGrid(amplitude, gridSnap)
    return amplitude, phase

# This is the nutty one
def spectrogramToAudioFile(spectrogram, fftWindowSize=2048, phaseIterations=10, phase=None):
    if phase is not None:
        # reconstructing the new complex matrix
        squaredAmplitudeAndSquaredPhase = np.power(spectrogram, 2)
        squaredPhase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase - squaredPhase,0))
        amplitude = np.expm1(unexpd)
        stftMatrix = amplitude + phase * 1j
        audio = librosa.istft(stftMatrix)
    else:
        # Phase reconstruction successive approximation wizardry
        # credit to https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/
        # although they do *way* more iterations than you need (10 is fine, 20 would be better, 500 is overkill)
        amplitude = np.zeros_like(spectrogram)
        amplitude[:spectrogram.shape[0],:] = np.exp(spectrogram) - 1
        phase = 2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi
        for i in range(phaseIterations):
            spectrum = amplitude * np.exp(1j*phase)
            audio = librosa.istft(spectrum)
            p = np.angle(librosa.stft(audio, fftWindowSize))
    return audio

def loadSpectrogram(filePath):
    fileName = basename(filePath)
    if filePath.index("sampleRate") < 0:
        console.warn("sample rate not specified")
    sampleRate = int(fileName[fileName.index("sampleRate=") + 11:fileName.index(").png")])
    console.info("Sample rate : " + str(sampleRate))
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

    sanityCheck, phase = audioFileToSpectrogram(audio)
    saveSpectrogram(sanityCheck, fileName + fileSuffix("Output Spectrogram", fft=args.fft, iter=args.iter, sampleRate=sampleRate) + ".png")

    saveAudioFile(audio, fileName + fileSuffix("Output", fft=args.fft, iter=args.iter) + ".wav", sampleRate)

if __name__ == "__main__":
    # for experimenting with modifying acapellas in image processors :)
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
