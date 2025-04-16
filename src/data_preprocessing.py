## 📁 nmf_source_separation/src/preprocessing.py
import librosa
import numpy as np

def load_and_prepare_audio(path1, path2, path3, trim=True):
    """
    Load three audio files, resample to match sampling rate,
    and trim to the shortest length.
    Returns a dictionary of aligned numpy arrays.
    """
    y1, sr1 = librosa.load(path1, sr=None, mono=True)
    y2, sr2 = librosa.load(path2, sr=None, mono=True)
    y3, sr3 = librosa.load(path3, sr=None, mono=True)

    y1_resampled = librosa.resample(y1, orig_sr=sr1, target_sr=sr2)

    if trim:
        y1_trimmed = y1_resampled[sr2*50:]

    audio_dict = {
    "Sing": y1_trimmed,
    "B-N": y2,
    "B-IR": y3,
    "SR": sr2   
    }
    return audio_dict

def train_test(audio_dict):
    voice = audio_dict["Sing"]
    bearing_N = audio_dict["B-N"]
    bearing_IR = audio_dict["B-IR"]
    sr = audio_dict["SR"]

    train_voice = voice[:sr*15]
    test_voice = voice[sr*15:sr*21]

    train_bearing = bearing_N[:sr*15]
    test_bearing = bearing_IR[sr*3:sr*9]

    train_mixture = train_voice + train_bearing
    test_mixture = test_voice + test_bearing

    return train_voice, test_voice, train_bearing, test_bearing, train_mixture, test_mixture


