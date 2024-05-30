import librosa
import matplotlib.pyplot as plt

def extract_beat_timing(musicpath):
    # Load an audio file
    y, sr = librosa.load(musicpath, sr=None)  # Load with native sampling rate

    # Perform beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return (tempo, beat_times)