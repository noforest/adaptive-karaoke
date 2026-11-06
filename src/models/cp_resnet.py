from pathlib import Path
import re

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader


# ---- Data ----

def tokenize(text:str):
    return re.findall(r"\b\w+\b", text.lower())


class AudioLyricsDataset(Dataset):
    def __init__(self, rows, vocab, label_to_id, sample_rate=22050):
        """
        rows: list of dicts, e.g.
          {"audio": "data/song1.wav", "lyrics": "data/song1.txt",
           "label": "class_a"}
        """
        self.rows = rows
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.sample_rate = sample_rate

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        waveform, sr = torchaudio.load(row["audio"])
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )

        # Shape: [1, mel_bins, time]
        spec = self.to_db(self.mel(waveform))

        text = Path(row["lyrics"]).read_text(encoding="utf-8")
        tokens = tokenize(text)
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        if not ids:
            ids = [self.vocab["<unk>"]]

        return spec, torch.tensor(ids), self.label_to_id[row["label"]]


def collate_batch(batch):
    specs, texts, labels = zip(*batch)

    # Pad time dimension of spectrograms to the longest clip in the batch.
    max_t = max(x.shape[-1] for x in specs)
    padded_specs = torch.stack([
        nn.functional.pad(x, (0, max_t - x.shape[-1]))
        for x in specs
    ])

    # Pad lyric token sequences.
    lengths = torch.tensor([len(x) for x in texts])
    padded_texts = nn.utils.rnn.pad_sequence(
        texts, batch_first=True, padding_value=0
    )

    return padded_specs, padded_texts, lengths, torch.tensor(labels)


# ---- CP-ResNet-style audio branch ----

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        residual = self.skip(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class AudioEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, spec):
        return self.net(spec)


class LyricsEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, out_dim=128, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, emb_dim, padding_idx=pad_id
        )
        self.rnn = nn.GRU(emb_dim, out_dim, batch_first=True)

    def forward(self, tokens, lengths):
        x = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        return h[-1]


class AudioLyricsModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.audio_encoder = AudioEncoder(out_dim=128)
        self.lyrics_encoder = LyricsEncoder(vocab_size, out_dim=128)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, spec, tokens, lengths):
        audio_features = self.audio_encoder(spec)
        lyrics_features = self.lyrics_encoder(tokens, lengths)
        combined = torch.cat([audio_features, lyrics_features], dim=1)
        return self.classifier(combined)


# ---- Training step ----

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create vocab, label_to_id, rows, and dataset from your training metadata.
# Reserve 0 for padding and include an unknown-token entry.
#
# dataset = AudioLyricsDataset(rows, vocab, label_to_id)
# loader = DataLoader(dataset, batch_size=16, shuffle=True,
#                     collate_fn=collate_batch)
# model = AudioLyricsModel(len(vocab), num_classes=len(label_to_id)).to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# loss_fn = nn.CrossEntropyLoss()
#
# for specs, tokens, lengths, labels in loader:
#     specs, tokens, labels = specs.to(device), tokens.to(device), labels.to(device)
#     logits = model(specs, tokens, lengths)
#     loss = loss_fn(logits, labels)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
