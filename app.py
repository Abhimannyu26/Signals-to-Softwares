import os
import io
import csv
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

MUSIC_DIR = os.environ.get("SONG_DIR", "songs")
INDEX_PATH = "db.pkl"

st.set_page_config(page_title="Sonic Signatures", layout="wide")


@st.cache_resource
def fetch_db():
    if os.path.exists(INDEX_PATH):
        return fp.load_database(INDEX_PATH)
    if not fp.list_audio(MUSIC_DIR):
        return {"db": {}, "names": [], "mode": "pairs"}
    song_db = fp.build_database(MUSIC_DIR, mode="pairs")
    fp.save_database(song_db, INDEX_PATH)
    return song_db


def load_clip(upload):
    import librosa
    raw = upload.read()
    wave, _ = librosa.load(io.BytesIO(raw), sr=fp.SR, mono=True)
    return wave


def compute_offsets(wave, song_db):
    qh = fp.fingerprint(wave, mode=song_db["mode"])
    offsets = defaultdict(Counter)
    for h, tq in qh:
        for idx, ts in song_db["db"].get(h, ()):
            offsets[song_db["names"][idx]][ts - tq] += 1
    return offsets, len(qh)


def draw_spectrogram(wave):
    S, freq_axis, time_axis = fp.spectrogram(wave)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.pcolormesh(time_axis, freq_axis, fp.spectrogram_db(S),
                  cmap="magma", vmin=-80, vmax=0, shading="auto")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram of the query clip")
    fig.tight_layout()
    return fig


def draw_constellation(wave):
    S, freq_axis, time_axis = fp.spectrogram(wave)
    ti, fi = fp.find_peaks(S)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.pcolormesh(time_axis, freq_axis, fp.spectrogram_db(S),
                  cmap="magma", vmin=-80, vmax=0, shading="auto")
    ax.scatter(time_axis[ti], freq_axis[fi], s=12, facecolors="none",
               edgecolors="cyan", linewidths=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Constellation (%d peaks)" % len(ti))
    fig.tight_layout()
    return fig


def draw_offsets(offsets, pred):
    fig, ax = plt.subplots(figsize=(7, 3))
    if pred and pred in offsets:
        h = offsets[pred]
        ax.vlines(list(h.keys()), 0, list(h.values()), color="tab:green", linewidth=1.3)
    ax.set_xlabel("time offset (frames)")
    ax.set_ylabel("matching hashes")
    ax.set_title("Offset histogram for the predicted song")
    fig.tight_layout()
    return fig


st.title("Sonic Signatures - audio fingerprint identifier")

song_db = fetch_db()
st.caption("Indexed library: %d songs - %s"
           % (len(song_db["names"]), ", ".join(song_db["names"]) if song_db["names"] else "(empty)"))

if not song_db["names"]:
    st.error("No songs are indexed. Set SONG_DIR to the song folder, or ship a db.pkl.")
    st.stop()

ui_mode = st.sidebar.radio("Mode", ["Single clip", "Batch"])

if ui_mode == "Single clip":
    upload = st.file_uploader("Upload a query clip",
                              type=["wav", "mp3", "flac", "ogg", "m4a"])
    if upload is not None:
        wave = load_clip(upload)
        st.audio(upload)

        pred, match_scores, _ = fp.match(wave, song_db)
        if pred is None:
            st.warning("No match found - the clip produced no overlapping hashes.")
        else:
            ranking = sorted(match_scores.items(), key=lambda kv: -kv[1])
            runner_up = ranking[1][1] if len(ranking) > 1 else 0
            st.success("Prediction: **%s**  (score %d)" % (pred, match_scores[pred]))
            st.write("Score margin over the runner-up: %d vs %d" % (match_scores[pred], runner_up))

        offsets, nhashes = compute_offsets(wave, song_db)
        left, right = st.columns(2)
        with left:
            st.pyplot(draw_spectrogram(wave))
            st.pyplot(draw_offsets(offsets, pred))
        with right:
            st.pyplot(draw_constellation(wave))
            if match_scores:
                st.write("Top candidates")
                st.table({"song": [r[0] for r in ranking[:5]],
                          "score": [r[1] for r in ranking[:5]]})

elif ui_mode == "Batch":
    st.write("Upload several clips. The app writes `results.csv` with two "
             "columns: `filename`, `prediction` (filename without extension).")
    uploads = st.file_uploader("Upload query clips",
                               type=["wav", "mp3", "flac", "ogg", "m4a"],
                               accept_multiple_files=True)
    if uploads:
        csv_rows = []
        bar = st.progress(0.0)
        for i, upload in enumerate(uploads):
            wave = load_clip(upload)
            pred, _, _ = fp.match(wave, song_db)
            clip_name = os.path.splitext(os.path.basename(upload.name))[0]
            csv_rows.append((clip_name, pred if pred is not None else ""))
            bar.progress((i + 1) / len(uploads))

        st.table({"filename": [r[0] for r in csv_rows],
                  "prediction": [r[1] for r in csv_rows]})

        out_buf = io.StringIO()
        writer = csv.writer(out_buf)
        writer.writerow(["filename", "prediction"])
        writer.writerows(csv_rows)
        st.download_button("Download results.csv", out_buf.getvalue(),
                           file_name="results.csv", mime="text/csv")
