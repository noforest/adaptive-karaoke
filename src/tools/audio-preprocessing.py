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

    - sr=16000 : 16 000 échantillons/s, fréquences jusqu'à 8 kHz. Pourquoi : la
      voix tient sous 8 kHz, c'est le standard parole (Whisper, wav2vec) et ~3x
      moins de données qu'en 44,1 kHz.
    - n_fft=512, soit 32 ms : 512 valeurs par cloche, taille de la fenêtre
      d'audio pour calculer une colonne. Pourquoi : la voix est quasi stable
      sur 20-40 ms, donc bon compromis précision fréquence / temps.
    - hop_length=160, soit 10 ms : décalage entre deux colonnes, donc 100
      colonnes/s. 
    - n_mels=80 : nombre de bandes de fréquence gardées, espacées comme
      l'oreille (fines en grave, larges en aigu). Pourquoi : compresse les 257
      bins FFT en 80 sans perdre l'info utile, standard (Whisper, Tacotron).
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
