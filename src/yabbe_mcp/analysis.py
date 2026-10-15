"""Helpers that let an AI "see" and "hear" media: contact sheets of frames, and a
rough reading of a sound's loudness, silences, onsets and tempo (from the peak
envelope YABBE computes for its waveforms: 100 peaks a second, 0..255)."""

from __future__ import annotations

import io
import math
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

PEAKS_PER_SECOND = 100


# -- pictures --------------------------------------------------------------------------------

def contact_sheet(frames: list[tuple[bytes, str]], columns: int = 0, width: int = 1280) -> bytes:
    """Tile JPEG/PNG frames into one labelled JPEG (fewer tokens than many images)."""
    images = [Image.open(io.BytesIO(b)).convert("RGB") for b, _ in frames]
    if not images:
        raise ValueError("no frames")
    n = len(images)
    cols = columns or (1 if n == 1 else 2 if n <= 4 else 3 if n <= 9 else 4)
    rows = math.ceil(n / cols)
    cell_w = width // cols
    ar = images[0].height / max(1, images[0].width)
    cell_h = int(cell_w * ar)
    label_h = 22
    sheet = Image.new("RGB", (cell_w * cols, (cell_h + label_h) * rows), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(16)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    for i, (im, (_, label)) in enumerate(zip(images, frames)):
        x, y = (i % cols) * cell_w, (i // cols) * (cell_h + label_h)
        im.thumbnail((cell_w - 4, cell_h - 4))
        sheet.paste(im, (x + (cell_w - im.width) // 2, y + (cell_h - im.height) // 2))
        draw.text((x + 6, y + cell_h + 2), label, fill=(235, 235, 240), font=font)
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=82)
    return out.getvalue()


def side_by_side(a: bytes, b: bytes, labels: tuple[str, str]) -> bytes:
    return contact_sheet([(a, labels[0]), (b, labels[1])], columns=2)


def overlay_mask(frame: bytes, mask_png: bytes, color=(255, 40, 140)) -> bytes:
    """The frame with the selected area tinted (to check a roto selection)."""
    im = Image.open(io.BytesIO(frame)).convert("RGB")
    m = Image.open(io.BytesIO(mask_png)).convert("L").resize(im.size)
    tint = Image.new("RGB", im.size, color)
    mixed = Image.composite(Image.blend(im, tint, 0.55), im, m)
    out = io.BytesIO()
    mixed.save(out, "JPEG", quality=85)
    return out.getvalue()


# -- sound ----------------------------------------------------------------------------------------

def analyse_peaks(peaks: bytes, start: float = 0.0, end: Optional[float] = None,
                  silence_level: float = 0.06, min_silence: float = 0.4) -> dict:
    """Loudness over time, silent stretches, strong onsets and an estimated tempo."""
    env = [p / 255 for p in peaks]
    i0 = max(0, int(start * PEAKS_PER_SECOND))
    i1 = len(env) if end is None else min(len(env), int(end * PEAKS_PER_SECOND))
    env = env[i0:i1]
    if not env:
        return {"error": "no sound in that range"}
    t = lambda i: round(start + i / PEAKS_PER_SECOND, 2)  # noqa: E731

    # loudness every half second (0 = silent, 1 = the loudest moment of the file)
    step = PEAKS_PER_SECOND // 2
    loud = [round(sum(env[i:i + step]) / len(env[i:i + step]), 2) for i in range(0, len(env), step)]

    silences = []
    run = None
    for i, v in enumerate(env + [1.0]):
        if v < silence_level:
            run = i if run is None else run
        elif run is not None:
            if (i - run) / PEAKS_PER_SECOND >= min_silence:
                silences.append([t(run), t(i)])
            run = None

    # onsets: sudden rises of the envelope
    novelty = [0.0] * len(env)
    for i in range(5, len(env)):
        prev = sum(env[i - 5:i]) / 5
        novelty[i] = max(0.0, env[i] - prev)
    thr = max(0.08, sorted(novelty)[int(len(novelty) * 0.97)] if len(novelty) > 30 else 0.1)
    onsets = []
    last = -1e9
    for i in range(1, len(novelty) - 1):
        if novelty[i] >= thr and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1] and i - last > 12:
            onsets.append(t(i))
            last = i

    tempo = _tempo(novelty)
    out = {
        "range": [t(0), t(len(env))],
        "loudness_every_0_5s": loud if len(loud) <= 600 else None,
        "silences": silences[:200],
        "onsets": onsets[:400],
    }
    if tempo:
        bpm, phase = tempo
        period = 60 / bpm
        beats = []
        b = phase / PEAKS_PER_SECOND
        while b < len(env) / PEAKS_PER_SECOND and len(beats) < 800:
            beats.append(round(start + b, 2))
            b += period
        out["tempo_bpm"] = round(bpm, 1)
        out["beats"] = beats
    if out["loudness_every_0_5s"] is None:
        out["note"] = "Long file: loudness omitted, ask for a shorter range"
    return out


def _tempo(novelty: list[float]) -> Optional[tuple[float, int]]:
    """Tempo from the autocorrelation of the onset curve (60-190 bpm), with
    octave correction (a beat at twice the speed that fits nearly as well wins)."""
    n = len(novelty)
    if n < PEAKS_PER_SECOND * 4:
        return None
    # widen each onset a little so off-by-a-frame hits still line up
    smooth = [max(novelty[max(0, i - 2):i + 3]) for i in range(n)]
    mean = sum(smooth) / n
    x = [v - mean for v in smooth]
    lo, hi = int(PEAKS_PER_SECOND * 60 / 190), int(PEAKS_PER_SECOND * 60 / 60)
    score = {}
    for lag in range(lo, hi + 1):
        score[lag] = sum(x[i] * x[i + lag] for i in range(n - lag)) / (n - lag)
    best_lag = max(score, key=score.get)
    if score[best_lag] <= 0:
        return None
    half = round(best_lag / 2)
    if half >= lo and score.get(half, 0) >= 0.8 * score[best_lag]:
        best_lag = half
    phase = max(range(best_lag), key=lambda p: sum(novelty[p::best_lag]))
    return 60 * PEAKS_PER_SECOND / best_lag, phase
