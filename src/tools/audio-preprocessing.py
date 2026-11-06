import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SONGS_DIR = DATA_DIR / "songs"
SPECTRO_DIR = DATA_DIR / "spectro"

def wav_to_spectro():
    SPECTRO_DIR.mkdir(exist_ok=True)
    songs_path = os.listdir(SONGS_DIR)
    for song in songs_path:
        if not song.endswith(".wav"):
            continue
        spectro_path = SPECTRO_DIR / f"{Path(song).stem}.png"
        if not spectro_path.exists():
            y, sr = librosa.load(SONGS_DIR / song)
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='inferno')

            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram of Audio File')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.tight_layout()
            plt.savefig(spectro_path)
            plt.close()


if __name__ == "__main__":
    wav_to_spectro()
