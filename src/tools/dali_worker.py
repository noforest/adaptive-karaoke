import argparse, random, time, sys
from pathlib import Path
import dali_reader as dr

sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument('--start', type=int, required=True)
ap.add_argument('--end', type=int, required=True)
ap.add_argument('--reverse', action='store_true')
args = ap.parse_args()

# Découpe uniquement une tranche [start:end]
ids = dr.list_ids()[args.start:args.end]
if args.reverse:
    ids = list(reversed(ids))

for sid in ids:
    target = dr.SONGS_DIR / f'{sid}.wav'
    if target.exists():
          # Saute si le .wav existe déjà
        continue
    annot = dr.read_annotation(sid)
    # Pause aléatoire entre 2 et 4 secondes après un téléchargement réussi
    if dr.download_song(annot, dr.SONGS_DIR):
        time.sleep(random.uniform(2.0, 4.0))


# Worker 1 (Tranche 0 -> 2600) :
#   nohup python -u src/tools/dali_worker.py --start 0 --end 2600 > worker1.log 2>&1 &

# Worker 2 (Tranche 2600 -> 5200) :
#   nohup python -u src/tools/dali_worker.py --start 2600 --end 5200 > worker2.log 2>&1 &

# Worker 3 (Tranche 5200 -> 7756, en partant de la fin) :
#   nohup python -u src/tools/dali_worker.py --start 5200 --end 7756 --reverse > worker3.log 2>&1 &

