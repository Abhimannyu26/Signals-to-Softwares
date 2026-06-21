"""
Q3B - Signals to Softwares
Interactive music identifier built on the Q3A fingerprinting pipeline.

Two modes:
  * Single clip   : upload one clip, see the spectrogram, the constellation,
                    the offset histogram and the prediction.
  * Batch         : upload several clips, get a results.csv with exactly two
                    columns (filename, prediction); filename has no extension.

The song database is indexed once and cached. A pre-built index (db.pkl) is
shipped with the app; if it is missing the app indexes everything under
SONG_DIR on first run.
"""

import os
import io
import csv
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

SONG_DIR = os.environ.get("SONG_DIR", "songs")
DB_PATH = "db.pkl"

st.set_page_config(page_title="Sonic Signatures", layout="wide")


# ---------------------------------------------------------------------------
# Database (built once, cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_database():
    if os.path.exists(DB_PATH):
        return fp.load_database(DB_PATH)
    if not fp.list_audio(SONG_DIR):
        return {"db": {}, "names": [], "mode": "pairs"}
    db = fp.build_database(SONG_DIR, mode="pairs")
    fp.save_database(db, DB_PATH)
    return db


def read_upload(uploaded):
    """Decode an uploaded audio file to a mono waveform at fp.SR."""
    import librosa
    data = uploaded.read()
    y, _ = librosa.load(io.BytesIO(data), sr=fp.SR, mono=True)
    return y


def offsets_for(query_y, database):
    """Per-song offset histograms for a query (for plotting)."""
    qh = fp.fingerprint(query_y, mode=database["mode"])
    per = defaultdict(Counter)
    for h, tq in qh:
        for sid, tdb in database["db"].get(h, ()):
            per[database["names"][sid]][tdb - tq] += 1
    return per, len(qh)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_spectrogram(y):
    S, freqs, times = fp.spectrogram(y)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.pcolormesh(times, freqs, fp.spectrogram_db(S),
                  cmap="magma", vmin=-80, vmax=0, shading="auto")
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram of the query clip")
    fig.tight_layout()
    return fig


def plot_constellation(y):
    S, freqs, times = fp.spectrogram(y)
    t_idx, f_idx = fp.find_peaks(S)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.pcolormesh(times, freqs, fp.spectrogram_db(S),
                  cmap="magma", vmin=-80, vmax=0, shading="auto")
    ax.scatter(times[t_idx], freqs[f_idx], s=12, facecolors="none",
               edgecolors="cyan", linewidths=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("Constellation (%d peaks)" % len(t_idx))
    fig.tight_layout()
    return fig


def plot_offset_hist(per, best):
    fig, ax = plt.subplots(figsize=(7, 3))
    if best and best in per:
        h = per[best]
        ax.bar(list(h.keys()), list(h.values()), width=1.0, color="tab:green")
    ax.set_xlabel("time offset (frames)")
    ax.set_ylabel("matching hashes")
    ax.set_title("Offset histogram for the predicted song")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Sonic Signatures - audio fingerprint identifier")

db = get_database()
st.caption("Indexed library: %d songs - %s"
           % (len(db["names"]), ", ".join(db["names"]) if db["names"] else "(empty)"))

if not db["names"]:
    st.error("No songs are indexed. Set SONG_DIR to the song folder, or ship a db.pkl.")
    st.stop()

mode = st.sidebar.radio("Mode", ["Single clip", "Batch"])

if mode == "Single clip":
    up = st.file_uploader("Upload a query clip",
                          type=["wav", "mp3", "flac", "ogg", "m4a"])
    if up is not None:
        y = read_upload(up)
        st.audio(up)

        best, scores, _ = fp.match(y, db)
        if best is None:
            st.warning("No match found - the clip produced no overlapping hashes.")
        else:
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            runner = ranked[1][1] if len(ranked) > 1 else 0
            st.success("Prediction: **%s**  (score %d)" % (best, scores[best]))
            st.write("Score margin over the runner-up: %d vs %d"
                     % (scores[best], runner))

        per, nq = offsets_for(y, db)
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_spectrogram(y))
            st.pyplot(plot_offset_hist(per, best))
        with c2:
            st.pyplot(plot_constellation(y))
            if scores:
                st.write("Top candidates")
                st.table({"song": [r[0] for r in ranked[:5]],
                          "score": [r[1] for r in ranked[:5]]})

elif mode == "Batch":
    st.write("Upload several clips. The app writes `results.csv` with two "
             "columns: `filename`, `prediction` (filename without extension).")
    ups = st.file_uploader("Upload query clips",
                           type=["wav", "mp3", "flac", "ogg", "m4a"],
                           accept_multiple_files=True)
    if ups:
        rows = []
        prog = st.progress(0.0)
        for i, up in enumerate(ups):
            y = read_upload(up)
            best, _, _ = fp.match(y, db)
            stem = os.path.splitext(os.path.basename(up.name))[0]
            rows.append((stem, best if best is not None else ""))
            prog.progress((i + 1) / len(ups))

        st.table({"filename": [r[0] for r in rows],
                  "prediction": [r[1] for r in rows]})

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["filename", "prediction"])
        w.writerows(rows)
        st.download_button("Download results.csv", buf.getvalue(),
                           file_name="results.csv", mime="text/csv")
