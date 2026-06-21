"""
Audio fingerprinting for music identification (EE200 Q3).

A small Shazam-style identifier:
  audio -> spectrogram -> constellation of peaks -> paired hashes -> database,
and matching by a time-offset histogram.

The same functions are used by the analysis notebook and by the Streamlit app.
"""

import os
import glob
import pickle
from collections import defaultdict, Counter

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SR = 11025          # working sample rate (low rate keeps the informative band)
N_FFT = 1024        # STFT window length (~93 ms at 11025 Hz)
HOP = 512           # STFT hop (~46 ms)

# peak picking
PEAK_NEIGH_F = 10   # half-neighbourhood in frequency bins
PEAK_NEIGH_T = 10   # half-neighbourhood in time frames
PEAK_THRESH_DB = -65  # keep peaks louder than this (ref = spectrogram max = 0 dB)

# pairing peaks into hashes
FAN_OUT = 8         # how many later peaks an anchor is paired with
DT_MIN = 1          # minimum time gap (frames) between anchor and target
DT_MAX = 60         # maximum time gap (frames)
F_BAND = 80         # only pair peaks within this many frequency bins

AUDIO_EXTS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------
def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def spectrogram(y, n_fft=N_FFT, hop=HOP):
    """Magnitude STFT plus the matching frequency and time axes."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=hop)
    return S, freqs, times


def spectrogram_db(S):
    return librosa.amplitude_to_db(S, ref=np.max)


# ---------------------------------------------------------------------------
# Constellation (peak picking)
# ---------------------------------------------------------------------------
def find_peaks(S, neigh_f=PEAK_NEIGH_F, neigh_t=PEAK_NEIGH_T, thresh_db=PEAK_THRESH_DB):
    """Local maxima of the dB spectrogram that stand out from their neighbourhood.

    Returns arrays (t_idx, f_idx) of frame and frequency-bin indices, sorted by time.
    """
    S_db = spectrogram_db(S)
    local_max = maximum_filter(S_db, size=(2 * neigh_f + 1, 2 * neigh_t + 1)) == S_db
    keep = local_max & (S_db > thresh_db)
    f_idx, t_idx = np.where(keep)            # S is [freq, time]
    order = np.argsort(t_idx)
    return t_idx[order], f_idx[order]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def paired_hashes(t_idx, f_idx, fan_out=FAN_OUT):
    """Pair each anchor peak with a few later peaks.

    Yields (hash_key, anchor_time) where hash_key = (f1, f2, dt).
    """
    out = []
    n = len(t_idx)
    for i in range(n):
        t1, f1 = t_idx[i], f_idx[i]
        paired = 0
        for j in range(i + 1, n):
            dt = t_idx[j] - t1
            if dt < DT_MIN:
                continue
            if dt > DT_MAX:
                break
            if abs(int(f_idx[j]) - int(f1)) > F_BAND:
                continue
            out.append(((int(f1), int(f_idx[j]), int(dt)), int(t1)))
            paired += 1
            if paired >= fan_out:
                break
    return out


def single_hashes(t_idx, f_idx):
    """Degenerate fingerprint: each peak is its own hash (just the frequency bin)."""
    return [((int(f),), int(t)) for t, f in zip(t_idx, f_idx)]


def fingerprint(y, mode="pairs"):
    S, _, _ = spectrogram(y)
    t_idx, f_idx = find_peaks(S)
    if mode == "pairs":
        return paired_hashes(t_idx, f_idx)
    return single_hashes(t_idx, f_idx)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def list_audio(song_dir):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(song_dir, ext)))
    return sorted(files)


def path_for(song_dir, name):
    """Resolve a song name (filename without extension) to its actual path,
    regardless of audio extension (.wav, .mp3, ...)."""
    for p in list_audio(song_dir):
        if os.path.splitext(os.path.basename(p))[0] == name:
            return p
    raise FileNotFoundError("%r not found in %r" % (name, song_dir))


def build_database(song_dir, mode="pairs"):
    """Index every song under song_dir into a hash -> [(song_id, t_anchor)] table."""
    db = defaultdict(list)
    names = []
    for sid, path in enumerate(list_audio(song_dir)):
        name = os.path.splitext(os.path.basename(path))[0]
        names.append(name)
        y = load_audio(path)
        for h, t in fingerprint(y, mode=mode):
            db[h].append((sid, t))
    return {"db": dict(db), "names": names, "mode": mode}


def save_database(database, path):
    with open(path, "wb") as f:
        pickle.dump(database, f)


def load_database(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match(query_y, database):
    """Identify query_y against an indexed database.

    Returns (best_name, scores, offsets) where scores[name] is the height of the
    best-aligned offset bin and offsets is the offset histogram for the winner.
    """
    db = database["db"]
    names = database["names"]
    mode = database["mode"]

    q = fingerprint(query_y, mode=mode)
    per_song = defaultdict(Counter)          # song_id -> Counter(offset -> votes)
    for h, t_q in q:
        for sid, t_db in db.get(h, ()):
            per_song[sid][t_db - t_q] += 1

    scores = {}
    best_offsets = {}
    for sid, hist in per_song.items():
        offset, votes = hist.most_common(1)[0]
        scores[names[sid]] = votes
        best_offsets[names[sid]] = hist

    if not scores:
        return None, {}, Counter()
    best = max(scores, key=scores.get)
    return best, scores, best_offsets[best]
