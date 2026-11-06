import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


SEGMENT_LENGTH = 30 # segment length in seconds



def make_slices(audio_path):

    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    samples_per_segment = SEGMENT_LENGTH * sr

    spectrograms = []

    for i in range(0, len(y), samples_per_segment):
        segment = y[i : i + samples_per_segment]

        # Ensure the segment is exactly 30s (pads with zeros if too short)
        if len(segment) < samples_per_segment:
            segment = np.pad(segment, (0, samples_per_segment - len(segment)))

        # Compute the STFT
        # n_fft is the window size; hop_length is the slide distance
        stft = librosa.stft(segment, n_fft=2048, hop_length=512)

        # Convert complex values to magnitude (absolute value)
        # Use librosa.amplitude_to_db to convert to decibels for better visualization/analysis
        magnitude = np.abs(stft)
        db_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)

        spectrograms.append(db_spectrogram)

    return spectrograms, sr


def plot_spectrogram(spec_to_plot, sampling_date):
    plt.figure(figsize=(10, 4))
    # sr is the sampling rate from librosa.load
    # x_axis='time' tells librosa to calculate the time stamps based on the sr
    librosa.display.specshow(spec_to_plot, sr=sampling_date, x_axis='time', y_axis='hz')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (30s Slice)')
    plt.tight_layout()
    plt.show()
