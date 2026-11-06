import librosa
import os
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SONGS_DIR = DATA_DIR / "songs"
SPECTRO_DIR = DATA_DIR / "spectro"

def wav_to_spectro():
    """
    Save a log-mel spectrogram (80, frames) per song, 1 frame = 10 ms.

    temps (ms)   0    10   20   30   40   50   60
                 |----|----|----|----|----|----|
    colonne 0    [=========32 ms=====]
    colonne 1         [=========32 ms=====]
    colonne 2              [=========32 ms=====]
    colonne 3                   [=========32 ms=====]
                 <-->
                 10 ms = hop_length

    - n_fft=512, soit 32 ms : chaque colonne du spectrogramme est calculée sur un morceau d'audio de 32 ms. Ça fixe la précision en fréquence.
    - hop_length=160, soit 10 ms : on décale la fenêtre de 10 ms pour calculer la colonne suivante. Ça fixe le nombre de colonnes par seconde, ici 100
    """
    SPECTRO_DIR.mkdir(exist_ok=True)
    songs_path = os.listdir(SONGS_DIR)
    for song in songs_path:
        if not song.endswith(".wav"):
            continue
        spectro_path = SPECTRO_DIR / f"{Path(song).stem}.npy"
        if not spectro_path.exists():
            y, sr = librosa.load(SONGS_DIR / song, sr=16000) #sample_rate=16000 (on peut représenter jusqu'à la moitié de sr donc 8Khz)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=80)
            S_db = librosa.power_to_db(S)
            np.save(spectro_path, S_db.astype(np.float32))


if __name__ == "__main__":
    wav_to_spectro()
