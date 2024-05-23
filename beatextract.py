import librosa
import matplotlib.pyplot as plt

# Load an audio file
y, sr = librosa.load('music\Ripple - Stuck [NCS Release].mp3', sr=None)  # Load with native sampling rate

# Perform beat tracking
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beats, sr=sr)

print("Estimated Tempo:", tempo, "beats per minute")
print("Beat Times in seconds:", beat_times)

# Plot the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.vlines(beat_times, -1, 1, color='r', alpha=0.9, label='Beats')

# Add labels and legend
plt.title('Waveform and Beat Times')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Display the plot
plt.show()