"""
Read DALI v2.0 annotations and f0 from the extracted folders in data/.
annot_tismir --> annotations
f0_tismir --> la mélodie de la voix
"""
import gzip
import pickle
from pathlib import Path
import youtube_dl

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ANNOT_DIR = DATA_DIR / "annot_tismir"
F0_DIR = DATA_DIR / "f0_tismir"
BASE_URL = "http://www.youtube.com/watch?v="

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



class MyLogger(object):
    def debug(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

def audio_from_url(url, name, path_output, errors=[]):
    """
    Download audio from a url.
        url : str
            url of the video (after watch?v= in youtube)
        name : str
            used to store the data
        path_output : str
            path for storing the data
    """
    error = None

    # ydl(youtube_dl.YoutubeDL): extractor
    ydl = get_my_ydl(path_output)

    ydl.params['outtmpl'] = ydl.params['outtmpl'] % {
        'ext': ydl.params['postprocessors'][0]['preferredcodec'],
        'title': name}

    if ydl:
        print ("Downloading " + url)
        try:
            ydl.download([BASE_URL + url])
        except Exception as e:
            print(e)
            error = e
    if error:
        errors.append([name, url, error])
    return

def get_my_ydl(directory:str):
    ydl = None
    outtmpl = None
    outtmpl = Path(directory +  f'/{id}.mp3')
    ydl_opts = {'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio',
                                    'preferredcodec': 'mp3',
                                    'preferredquality': '320'}],
                'outtmpl': outtmpl,
                'logger': MyLogger(),
                'progress_hooks': [my_hook],
                'verbose': False,
                'ignoreerrors': False,
                'external_downloader': 'ffmpeg',
                'nocheckcertificate': True}
                # 'external_downloader_args': "-j 8 -s 8 -x 8 -k 5M"}
                # 'maxBuffer': 'Infinity'}
                #  it uses multiple connections for speed up the downloading
                #  'external-downloader': 'ffmpeg'}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    ydl.cache.remove()
    import time
    time.sleep(.5)
    return ydl
def download_song(annot, filedir):
    id, url = annot['info']['id'], annot['info']['audio']['url']
    audio_from_url(url, id, filedir)


if __name__ == "__main__":
    song_id = list_ids()[0]
    annot = read_annotation(song_id)
    print(f"annot.keys() = {annot.keys()}\n")
    print(f"annot['info'].keys() = {annot['info'].keys()}\n")
    print(f"annot['annotations'].keys() = {annot['annotations'].keys()}\n")
    print(f"annot['errors'] = {annot['errors']}\n")

    print("========= Exemple =========")
    print(annot["info"]["artist"], "-", annot["info"]["title"])
    print(annot["annotations"]["annot"]["words"][0])
    print(read_f0(song_id)["f0"].shape)


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
