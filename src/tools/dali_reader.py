"""
Read DALI v2.0 annotations and f0 from the extracted folders in data/.
annot_tismir/<id>.gz --> annotations manuelles: paroles alignées dans le temps
    (notes, mots, lignes, paragraphes, phonèmes) + métadonnées du morceau
f0_tismir/<id>.f0.npz --> mélodie de la voix estimée par un modèle sur l'audio
    (72 demi-tons x frames), sert à vérifier l'alignement des annotations
"""
import gzip
import pickle
from pathlib import Path

import numpy as np
import yt_dlp

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ANNOT_DIR = DATA_DIR / "annot_tismir"
F0_DIR = DATA_DIR / "f0_tismir"
BASE_URL = "https://www.youtube.com/watch?v="

class Annotations:
    """Replaces DALI.Annotations so the DALI package is not needed."""


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "DALI.Annotations":
            return Annotations
        # Pickled with numpy 1.x: numpy.core became numpy._core in 2.x.
        return super().find_class(module.replace("numpy.core", "numpy._core"), name)


def list_ids():
    """Ids of all songs."""
    return sorted(path.name.removesuffix(".gz") for path in ANNOT_DIR.glob("*.gz"))


def read_annotation(song_id):
    """Dict with 'info', 'annotations' and 'errors' for one song."""
    with gzip.open(ANNOT_DIR / f"{song_id}.gz") as f:
        return vars(_Unpickler(f).load())


def read_f0(song_id):
    """Dict with 'f0' [72, frames], 'corr_vector', 'time_r' and 'freqs'."""
    with np.load(F0_DIR / f"{song_id}.f0.npz") as npz:
        return dict(npz)



def download_song(annot, out_dir):
    """Download the song audio as <out_dir>/<id>.mp3. Returns True if it worked."""
    song_id = annot["info"]["id"]
    url = BASE_URL + annot["info"]["audio"]["url"]
    opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/{song_id}.%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
        "quiet": True,
        "js_runtimes": {"node": {}},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        return True
    except yt_dlp.utils.DownloadError as e:
        print(f"{song_id}: {e}")
        return False


if __name__ == "__main__":
    # song_id = list_ids()[0]
    # annot = read_annotation(song_id)
    # print(f"annot.keys() = {annot.keys()}\n")
    # print(f"annot['info'].keys() = {annot['info'].keys()}\n")
    # print(f"annot['annotations'].keys() = {annot['annotations'].keys()}\n")
    # print(f"annot['errors'] = {annot['errors']}\n")
    #
    # print("========= Exemple =========")
    # print(annot["info"]["audio"], "-", annot["info"]["title"])
    # print(annot["info"]["artist"], "-", annot["info"]["title"])
    # print(annot["annotations"]["annot"]["words"][0])
    #
    # f0 = read_f0(song_id)
    # print(f"f0.keys() = {f0.keys()}")
    # print(f0["f0"].shape)

    songs_id = list_ids()
    for song_id in songs_id:
        annot = read_annotation(song_id)
        download_song(annot, "/home/nforest001/sync-enseirb/s9/_projet_semestriel/free-karaoke/data/songs")

     


    """
    annot
    ├── "info"                 → infos sur le morceau
    │   ├── "artist"           → "Francesco Renga"
    │   ├── "title"            → "Alba"
    │   ├── "audio"            → {"url": id YouTube, "path": ..., "working": ...}
    │   ├── "metadata"         → {"album", "release_date", "genres", "language", ...}
    │   └── "scores", "id", ...
    ├── "annotations"
    │   ├── "type"             → "horizontal"
    │   ├── "annot_param"      → {"fr": ..., "offset": ...}
    │   └── "annot"            → LES PAROLES
    │       ├── "notes"        → liste de segments
    │       ├── "words"        → liste de segments
    │       ├── "lines"        → liste de segments
    │       ├── "paragraphs"   → liste de segments
    │       └── "phonemes"     → liste de segments
    └── "errors"               → None
    """
