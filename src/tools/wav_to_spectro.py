import matplotlib.pyplot as plt
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
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    SPECTRO_DIR.mkdir(exist_ok=True)
    plt.savefig(SPECTRO_DIR / f"{Path(song_test).stem}.png")
    plt.show()

# for song in os.listdir(SONGS_DIR) :
#     sample_rate, samples = wavfile.read(song)
#     frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

