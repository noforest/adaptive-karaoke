import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """Conv2d with left-only padding along time.

    Input: [B, C, mel_bins, frames]
    Output has the same number of frames when time_stride=1.
    """
    def __init__(
        self, in_channels, out_channels,
        kernel_size=(3, 3), stride=(1, 1)
    ):
        super().__init__()
        self.time_kernel = kernel_size[1]
        self.time_stride = stride[1]
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, 0),  # frequency padding only
        )

    def forward(self, x):
        # Pad only on the left of the time axis.
        left_pad = self.time_kernel - 1
        x = F.pad(x, (left_pad, 0, 0, 0))
        return self.conv(x)


class AcousticCRNN(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_phones: int,
        lstm_hidden: int = 1024,
    ):
        super().__init__()

        # Example CNN feature extractor. Pool frequency only, preserving
        # one CNN output step per input frame.
        self.cnn = nn.Sequential(
            CausalConv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            CausalConv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            CausalConv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Calculate CNN feature size after the two frequency pools.
        reduced_mels = n_mels // 4
        cnn_feature_size = 128 * reduced_mels

        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output_layer = nn.Linear(lstm_hidden, n_phones)

    def forward(self, log_mel):
        """
        log_mel: [B, T, n_mels]
        returns logits: [B, T, n_phones]
        """
        x = log_mel.unsqueeze(1)       # [B, 1, T, M]
        x = x.transpose(2, 3)         # [B, 1, M, T]

        x = self.cnn(x)               # [B, C, M', T]
        x = x.permute(0, 3, 1, 2)    # [B, T, C, M']
        x = x.flatten(start_dim=2)    # [B, T, C*M']

        x, _ = self.lstm(x)           # [B, T, 1024]
        return self.output_layer(x)   # [B, T, n_phones]


# Example training step
device = "cuda" if torch.cuda.is_available() else "cpu"

n_mels = 80
n_phones = 40  # Include whatever phone inventory your labels use.
model = AcousticCRNN(n_mels, n_phones).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# log_mel: [B, T, n_mels], computed from audio
# phone_ids: [B, T], frame-aligned integer phone labels
#
# Use ignore_index for padded frames, if batches contain padding.
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# logits = model(log_mel.to(device))
# targets = phone_ids.to(device)
# loss = criterion(
#     logits.reshape(-1, n_phones),
#     targets.reshape(-1),
# )
#
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# Inference:
# logits = model(log_mel)
# ppg = torch.softmax(logits, dim=-1)  # [B, T, n_phones]
