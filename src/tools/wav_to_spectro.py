import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import os


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SONGS_DIR = DATA_DIR / "songs"
SPECTRO_DIR = DATA_DIR / "spectro"


if len(os.listdir(SONGS_DIR)) != 0:
    song_test = os.listdir(SONGS_DIR)[0]
    sample_rate, samples = wavfile.read(SONGS_DIR / song_test)
    if samples.ndim == 2:  # stereo -> mono
        samples = samples.mean(axis=1)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # dB relative to the loudest point: 0 dB = max, everything else is negative
    spectrogram_db = 10 * np.log10(spectrogram / spectrogram.max() + 1e-10)
    plt.imshow(spectrogram_db, aspect="auto", origin="lower", cmap="inferno",
               vmin=-80, vmax=0,
               extent=[times[0], times[-1], frequencies[0], frequencies[-1]])
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label="dB (relative to max)")

    SPECTRO_DIR.mkdir(exist_ok=True)
    plt.savefig(SPECTRO_DIR / f"{Path(song_test).stem}.png")
    plt.show()

# for song in os.listdir(SONGS_DIR) :
#     sample_rate, samples = wavfile.read(song)
#     frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

