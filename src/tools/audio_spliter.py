from pathlib import Path
import openunmix
import torch
import torchaudio

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
UMX_SR = 44100

def load_stereo_44k(path):
    """Charge un fichier audio en stéréo 44,1 kHz : tenseur [1, 2, T]."""
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    if sr != UMX_SR:
        wav = torchaudio.functional.resample(wav, sr, UMX_SR)
    return wav.unsqueeze(0)

def to_mono_sr(wav, sr_out):
    """[2, T] à 44,1 kHz -> [1, T'] mono à sr_out."""
    mono = wav.mean(dim=0, keepdim=True)
    if sr_out == UMX_SR:
        return mono
    return torchaudio.functional.resample(mono, UMX_SR, sr_out)

def separate(separator, audio, device):
    """Renvoie (voix, accompagnement), chacun de forme [2, T], sur CPU."""
    try:
        with torch.inference_mode():
            est = separator(audio.to(device))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        separator.to("cpu")
        with torch.inference_mode():
            est = separator(audio)
        separator.to(device)
    est = est.cpu()
    return est[0, 0], est[0, 1]

def separer_audio(in_dir=DATA_DIR / "songs", out_dir=DATA_DIR / "vocals", ext=".wav", suffix="_vocals", sr=44100, model="umxl", save_acc=True, cpu=False):
    """Sépare la voix et l'accompagnement de tous les fichiers d'un dossier."""
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    device = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    print(f"Modèle {model} sur {device}")

    loader = getattr(openunmix, model)
    separator = loader(targets=["vocals"], residual=True, device=device)
    separator.freeze()
    separator.to(device)

    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob(f"*{ext}"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier {ext} dans {in_dir}")

    failed = []
    for i, path in enumerate(files, 1):
        out_voc = out_dir / f"{path.stem}{suffix}.wav"
        if out_voc.exists():
            continue
        try:
            audio = load_stereo_44k(path)
            voc, acc = separate(separator, audio, device)
            torchaudio.save(str(out_voc), to_mono_sr(voc, sr), sr)
            if save_acc:
                out_acc = out_dir / f"{path.stem}_accompaniment.wav"
                torchaudio.save(str(out_acc), to_mono_sr(acc, sr), sr)
        except Exception as e:
            print(f"{path.name}: {e}")
            failed.append(path.name)
        if i % 20 == 0:
            print(f"{i}/{len(files)}")

    print(f"Terminé. Échecs : {len(failed)}")
    if failed:
        (out_dir / "separation_echecs.txt").write_text("\n".join(failed))

if __name__ == "__main__":
    # Lance la séparation avec les paramètres par défaut
    separer_audio()