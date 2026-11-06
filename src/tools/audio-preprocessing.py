import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# Load an audio file. Use your own file path.
audio_folder = '/home/lhenkel/Projects/free-karaoke/data/songs'
audio_path = audio_folder + '/' + os.listdir(audio_folder)[0]
y, sr = librosa.load(audio_path)

print(f"Audio length: {len(y)} samples")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(y)/sr:.2f} seconds")


D = librosa.stft(y)

# Convert the complex values to magnitude
S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

print(f"STFT shape: {D.shape}")
print(f"Magnitude Spectrogram shape: {S_db.shape}")

plt.figure(figsize=(10, 6))

# Display the spectrogram
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='viridis')

# Add a color bar to show amplitude scale
plt.colorbar(format='%+2.0f dB')

# Set the title and labels
plt.title('Spectrogram of Audio File')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

# Show the plot
plt.tight_layout()
plt.show()
