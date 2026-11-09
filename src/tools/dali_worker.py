import argparse, random, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dali_reader as dr

ap = argparse.ArgumentParser()
ap.add_argument('--start', type=int, required=True)
ap.add_argument('--end', type=int, required=True)
ap.add_argument('--reverse', action='store_true')
args = ap.parse_args()

# Découpe uniquement une tranche [start:end]
ids = dr.list_ids()[args.start:args.end]
if args.reverse:
    ids = list(reversed(ids))

# Vidéos privées/supprimées /bloquées: inutile de refaire une requête
skip = dr.read_unavailable()

for sid in ids:
    if sid in skip or (dr.SONGS_DIR / f'{sid}.wav').exists():
        continue
    annot = dr.read_annotation(sid)

    # Rate limit: on attend de plus en plus longtemps puis on retente le même son
    pause = 60
    while dr.download_song(annot, dr.SONGS_DIR) == 'rate_limited':
        print(f'{sid}: rate limited, pause {pause}s', flush=True)
        time.sleep(pause)
        pause = min(pause * 2, 1800)

    # Pause après chaque requête, échec compris (sinon les échecs s'enchaînent sans pause -> 429)
    time.sleep(random.uniform(2.0, 4.0))


# NOTE:
# nohup python -u src/tools/dali_worker.py --start 0 --end 7756 > worker_all.log 2>&1 &

# Worker 1 (Tranche 0 -> 2600) :
#   nohup python -u src/tools/dali_worker.py --start 0 --end 2600 > worker1.log 2>&1 &

# Worker 2 (Tranche 2600 -> 5200) :
#   nohup python -u src/tools/dali_worker.py --start 2600 --end 5200 > worker2.log 2>&1 &

# Worker 3 (Tranche 5200 -> 7756, en partant de la fin) :
#   nohup python -u src/tools/dali_worker.py --start 5200 --end 7756 --reverse > worker3.log 2>&1 &
