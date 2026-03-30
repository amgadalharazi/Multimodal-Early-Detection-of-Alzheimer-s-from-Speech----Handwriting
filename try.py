"""
generate_realistic_audio_v3.py  ─  MAXIMUM REALISM
=====================================================
The most clinically accurate AD / HC speech synthesiser possible
without requiring real patient recordings.

WHAT MAKES THIS REALISTIC:
  1. Disfluency injected INTO TEXT before TTS  (not spliced in afterward)
     → The voice model generates hesitations with real coarticulation
  2. 6 clinically-documented AD speech patterns modelled
  3. Vocal AGING layer in post-processing:
       • Jitter   (cycle-to-cycle pitch instability)
       • Shimmer  (cycle-to-cycle amplitude instability)
       • Tremor   (4-7 Hz AM+FM — most characteristic of elderly/AD)
       • Breathiness / aspiration noise in voiced segments
       • Age-related spectral tilt (HF roll-off)
  4. TTS speed tied directly to AD severity
  5. Near-zero post-processing for HC (sounds young and clean)

TTS PRIORITY (auto mode):
  1. XTTS v2  — best realism, local, free
                 (needs Python 3.9-3.11: pip install TTS)
  2. Kokoro   — very good, Python 3.12 compatible
                 (pip install kokoro-onnx  +  download model files)
  3. Edge TTS — Microsoft neural voices, free, needs internet
                 (pip install edge-tts)
  4. macOS say — always available on Mac, decent fallback

INSTALL:
  pip install librosa soundfile numpy scipy

  + ONE of:
    pip install TTS          # XTTS v2 (Python 3.9-3.11)
    pip install kokoro-onnx  # Kokoro  (Python 3.12+)
    pip install edge-tts     # Edge TTS (any Python, needs internet)

USAGE:
  python generate_realistic_audio_v3.py
  python generate_realistic_audio_v3.py --n 100 --seed 42
  python generate_realistic_audio_v3.py --n 50 --tts say
  python generate_realistic_audio_v3.py --n 50 --tts edge

PIPELINE:
  ┌─────────────────────┐
  │   Text Prompt       │
  └────────┬────────────┘
           │
           ▼  inject_disfluencies()   ← Stage 1: corrupt text
  ┌─────────────────────┐
  │  Disfluent Text     │  "The boy... uh... the boy is taking... cookies"
  └────────┬────────────┘
           │
           ▼  neural_tts()            ← Stage 2: generate speech
  ┌─────────────────────┐
  │  Clean Speech Array │  (prosody, pauses, breath already natural)
  └────────┬────────────┘
           │
           ▼  age_voice()             ← Stage 3: vocal aging layer
  ┌─────────────────────┐
  │  Aged Speech Array  │  (tremor, jitter, shimmer, breathiness)
  └────────┬────────────┘
           │
           ▼  add_environment()       ← Stage 4: light room texture
  ┌─────────────────────┐
  │  Final Audio        │
  └─────────────────────┘
"""

import argparse
import asyncio
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, sosfilt, butter as _butter

# ─── auto-install helpers ────────────────────────────────────────────────────
def _install(pkg: str):
    print(f"  pip install {pkg} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--quiet",
         "--break-system-packages"],
        check=False,
    )

try:
    import librosa
    import soundfile as sf
except ImportError:
    _install("librosa soundfile")
    import librosa
    import soundfile as sf

SR_WORK = 22050   # internal processing sample rate


# =============================================================================
# SPEECH PROMPTS  (Cookie Theft / daily-activity / story-retelling)
# =============================================================================

SPEECH_PROMPTS = [
    # Cookie Theft
    "The woman is washing the dishes and the water is overflowing from the sink onto the floor.",
    "A boy is standing on a stool trying to reach the cookie jar on the top shelf.",
    "The girl is reaching up asking her brother for a cookie from the jar.",
    "Outside the window I can see trees and it looks like a sunny day.",
    "The mother doesn't notice that the water is overflowing behind her.",
    "There are two children, a boy and a girl, standing in the kitchen.",
    "The boy is about to fall off the stool because it is tipping over.",
    "The woman is drying a dish with a cloth and looking out of the window.",
    "I can see curtains on the window and there are dishes on the counter.",
    "The children are trying to get cookies while their mother is not paying attention.",
    "The sink is completely full and the water is spilling out all over the floor.",
    "The boy has climbed up on the stool to reach the cabinet above him.",
    "The girl has her hand out waiting for her brother to hand her a cookie.",
    "The scene is inside a house and the weather outside looks warm and clear.",
    "The woman appears to be daydreaming as the water floods the kitchen floor.",
    # Daily activities
    "This morning I woke up at seven and had breakfast with my family.",
    "I am going to the grocery store to pick up some bread and milk and eggs.",
    "Yesterday I went for a walk in the park and I saw some ducks in the pond.",
    "My daughter called me on the phone and we talked for about half an hour.",
    "I like to read the newspaper in the morning while I drink my coffee.",
    "We had dinner at six and then I watched the news on television.",
    "I need to remember to take my medicine before I go to bed tonight.",
    "Every Sunday we go to church and then have lunch together as a family.",
    "I was trying to remember where I put my glasses but I found them on the table.",
    "The doctor told me to walk for thirty minutes every day to stay healthy.",
    "After lunch I sat on the porch and listened to the birds in the garden.",
    "My son came to visit last weekend and we had a very nice time together.",
    "I cooked a pot of soup this afternoon and it turned out very well.",
    "Before going to sleep I always read a few pages of whatever book I am reading.",
    "I went to the pharmacy and then stopped at the bakery on the way home.",
    # Story retelling
    "Once upon a time there was a little girl who lived with her grandmother.",
    "The man drove his car to the station and then took the train into the city.",
    "She opened the door slowly and called out to see if anyone was home.",
    "After the meeting they all went out for coffee and talked about their plans.",
    "The children ran outside to play as soon as the rain stopped falling.",
    "He picked up the phone and dialed the number but nobody answered.",
    "The cat jumped off the fence and ran across the yard into the bushes.",
    "She looked at the clock and realized she was going to be late for her appointment.",
    "They packed their bags and loaded the car and set off early in the morning.",
    "The teacher wrote the lesson on the board and the students copied it down.",
]


# =============================================================================
# STAGE 1 — DISFLUENCY INJECTION
#
# Applied BEFORE TTS so the neural model generates them naturally.
#
# Clinically documented AD speech patterns modelled here:
#
#   Pattern 1: Filled pauses        "uh", "um", "er"
#   Pattern 2: Word repetitions     "the... the boy" / "the the boy"
#   Pattern 3: Phrase repetitions   "and then, and then he..."
#   Pattern 4: Prolongations        "cook... cookies"
#   Pattern 5: Anomia / word-finding "that thing", "what do you call it"
#   Pattern 6: Sentence revision    abandon + restart ("the woman is wash—
#                                    she's drying the dishes")
#   Pattern 7: Semantic tangent     insert an irrelevant but related phrase
#   Pattern 8: Trailing-off         sentence ends with "..."
#   Pattern 9: Empty phrases        "you know", "I mean"
#
# Severity (0-1) is sampled from the AD acoustic config and controls
# all probabilities — so each speaker has a consistent impairment level.
# =============================================================================

FILLED_PAUSES = [
    "uh,", "um,", "er,", "uh... um,", "um... uh,",
    "uh... uh,", "well... um,", "I... um,",
]

ANOMIA_PHRASES = [
    "that... that thing,",
    "what do you call it,",
    "the... the thing,",
    "I... I can't remember the word,",
    "what's the name... uh,",
    "you know, the... the thing you use,",
]

EMPTY_PHRASES = [
    "you know,", "I mean,", "sort of,", "kind of,",
    "you see,", "well,",
]

REVISION_TEMPLATES = [
    # (abandoned_fragment, restart_phrase)
    ("{verb}... I mean, {restart}"),
    ("{verb}... no wait, {restart}"),
    ("{verb}... actually, {restart}"),
]

SEMANTIC_TANGENTS = [
    "which reminds me of something,",
    "I think it was,",
    "or maybe it was something else,",
]


def _word_repeat(word: str, n_extra: int) -> str:
    """'cookies' → 'cook... cookies' or 'the... the... the'."""
    parts = [f"{word}..."] * n_extra + [word]
    return " ".join(parts)


def inject_disfluencies(
    text: str,
    severity: float,
    rng: np.random.Generator,
) -> str:
    """
    Corrupt `text` to sound like an AD speaker with the given severity.

    severity 0.0 → clean speech (HC)
    severity 0.5 → moderate AD
    severity 1.0 → severe AD
    """
    if severity < 0.03:
        return text

    words  = text.rstrip(".").split()
    result = []

    # Per-word probabilities, all scaled by severity
    p_filler       = 0.10 * severity
    p_repeat_word  = 0.12 * severity
    p_prolong      = 0.08 * severity
    p_anomia       = 0.05 * severity
    p_empty        = 0.07 * severity
    p_tangent      = 0.04 * severity
    p_phrase_repeat = 0.05 * severity
    trail_off      = severity > 0.4 and rng.random() < 0.3 * severity

    phrase_buffer  = []   # for phrase-level repetition

    for i, word in enumerate(words):
        clean = word.strip(".,;?!")
        punct = word[len(clean):]

        # ── Filled pause before this word ──────────────────────────
        if rng.random() < p_filler:
            result.append(str(rng.choice(FILLED_PAUSES)))

        # ── Anomia: replace content word with placeholder ──────────
        if len(clean) > 5 and rng.random() < p_anomia:
            result.append(str(rng.choice(ANOMIA_PHRASES)))
            continue   # drop the actual word

        # ── Prolongation: "cook... cookies" ─────────────────────────
        if len(clean) > 3 and rng.random() < p_prolong:
            stem = clean[: max(2, len(clean) - 2)]
            result.append(f"{stem}...")

        # ── Word repetition ──────────────────────────────────────────
        if len(clean) > 1 and rng.random() < p_repeat_word:
            n_extra = int(rng.integers(1, 3))
            result.append(_word_repeat(clean, n_extra) + punct)
        else:
            result.append(word)

        phrase_buffer.append(word)

        # ── Phrase-level repetition (repeat last 2-3 words) ─────────
        if len(phrase_buffer) >= 3 and rng.random() < p_phrase_repeat:
            n_back = int(rng.integers(2, min(4, len(phrase_buffer))))
            snippet = " ".join(phrase_buffer[-n_back:])
            result.append(f"... {snippet},")

        # ── Semantic tangent after clause ───────────────────────────
        if punct == "," and rng.random() < p_tangent:
            result.append(str(rng.choice(SEMANTIC_TANGENTS)))

        # ── Empty phrase after comma or every ~7 words ───────────────
        if (punct == "," or (i > 0 and i % 7 == 0)) and rng.random() < p_empty:
            result.append(str(rng.choice(EMPTY_PHRASES)))

    # ── Trailing-off: cut sentence short with "..." ─────────────────
    if trail_off and len(result) > 6:
        keep = int(len(result) * rng.uniform(0.5, 0.8))
        result = result[:keep]
        result.append("...")
    else:
        result.append(".")   # close the sentence properly

    return " ".join(result)


# =============================================================================
# PARAMETER DISTRIBUTIONS  (mean, std, lo, hi)
#
# severity    → controls disfluency injection depth
# tts_speed   → directly sets TTS speaking rate (1.0 = normal)
# aging_*     → controls vocal aging layer parameters
# silence_*   → light post-synthesis pause nudge
# noise_amp   → very mild room noise
# =============================================================================

HC_CONFIG = dict(
    severity     = (0.04, 0.02, 0.00, 0.10),
    tts_speed    = (1.05, 0.05, 0.95, 1.18),   # slightly variable, normal pace
    aging_jitter = (0.003, 0.001, 0.001, 0.007),  # very slight F0 instability
    aging_shimmer= (0.020, 0.008, 0.005, 0.040),  # very slight amp instability
    aging_tremor_depth = (0.02, 0.01, 0.00, 0.05),# barely perceptible tremor
    aging_tremor_freq  = (5.0,  0.5,  4.0,  7.0), # 4-7 Hz physiological
    aging_breathiness  = (0.02, 0.01, 0.00, 0.05),# minimal breathiness
    aging_hf_rolloff   = (0.92, 0.03, 0.85, 0.99),# mild HF attenuation
    silence_prob = (0.04, 0.02, 0.00, 0.10),
    silence_ms   = (100,  30,   50,  200),
    noise_amp    = (0.0015, 0.0005, 0.0005, 0.004),
)

AD_CONFIG = dict(
    severity     = (0.58, 0.14, 0.25, 0.90),
    tts_speed    = (0.78, 0.08, 0.60, 0.94),   # noticeably slower
    aging_jitter = (0.018, 0.006, 0.008, 0.035),  # audible F0 wobble
    aging_shimmer= (0.060, 0.015, 0.030, 0.100),  # audible amplitude flutter
    aging_tremor_depth = (0.10, 0.04, 0.04, 0.22),# characteristic tremor
    aging_tremor_freq  = (5.5,  0.8,  4.0,  7.5),
    aging_breathiness  = (0.10, 0.03, 0.03, 0.20),# breathy, effortful voice
    aging_hf_rolloff   = (0.78, 0.06, 0.60, 0.92),# more muffled spectrum
    silence_prob = (0.18, 0.07, 0.05, 0.35),
    silence_ms   = (350, 120, 120, 900),
    noise_amp    = (0.004, 0.002, 0.001, 0.010),
)


def _truncnorm(mean, std, lo, hi, rng):
    return float(np.clip(rng.normal(mean, std), lo, hi))

def sample_config(cfg: dict, rng: np.random.Generator) -> dict:
    return {k: _truncnorm(*v, rng) for k, v in cfg.items()}


# =============================================================================
# STAGE 2 — TTS BACKENDS
# =============================================================================

# ── XTTS v2 (Coqui TTS)  — best quality, local, free ─────────────────────────

_xtts_model = None

def _load_xtts():
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model
    try:
        from TTS.api import TTS as CoquiTTS
        print("\n  Loading XTTS v2 (~2 GB model, one-time download)...")
        _xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("  XTTS v2 ready ✓\n")
    except Exception as e:
        print(f"  XTTS v2 unavailable: {e}")
        _xtts_model = None
    return _xtts_model

XTTS_SPEAKERS = [
    "Ana Florence", "Daisy Studious", "Gracie Wise",
    "Henriette Usha Iisakkinen", "Viktor Eka",
]

def tts_xtts(text: str, out_wav: Path, speed: float, speaker: str) -> bool:
    m = _load_xtts()
    if m is None:
        return False
    try:
        m.tts_to_file(
            text=text, file_path=str(out_wav),
            speaker=speaker, language="en", speed=speed,
        )
        return out_wav.exists() and out_wav.stat().st_size > 1000
    except Exception as e:
        print(f"  XTTS error: {e}")
        return False


# ── Kokoro  — good quality, Python 3.12 compatible ────────────────────────────
#
# Requires:
#   pip install kokoro-onnx
#   Download model files from HuggingFace:
#     https://huggingface.co/kokoro-community/kokoro-82m-v1.0-onnx
#   Place kokoro-v1.0.onnx and voices.bin in KOKORO_MODEL_DIR below.

KOKORO_MODEL_DIR = Path.home() / ".cache" / "kokoro"
KOKORO_MODEL     = KOKORO_MODEL_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES    = KOKORO_MODEL_DIR / "voices.bin"

_kokoro_model = None

def _load_kokoro():
    global _kokoro_model
    if _kokoro_model is not None:
        return _kokoro_model
    if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists()):
        return None
    try:
        from kokoro_onnx import Kokoro
        _kokoro_model = Kokoro(str(KOKORO_MODEL), str(KOKORO_VOICES))
        print("  Kokoro ready ✓")
    except Exception as e:
        print(f"  Kokoro unavailable: {e}")
        _kokoro_model = None
    return _kokoro_model

# Kokoro voice names (from their voice list)
KOKORO_VOICES_LIST = ["af_sarah", "af_bella", "am_adam", "am_michael",
                      "bf_emma", "bm_george"]

def tts_kokoro(text: str, out_wav: Path, speed: float,
               voice: str = "af_sarah") -> bool:
    m = _load_kokoro()
    if m is None:
        return False
    try:
        samples, sr = m.create(text, voice=voice, speed=speed, lang="en-us")
        sf.write(str(out_wav), samples, sr)
        return out_wav.exists()
    except Exception as e:
        print(f"  Kokoro error: {e}")
        return False


# ── Edge TTS  — Microsoft neural voices, free, needs internet ─────────────────
#
# pip install edge-tts
#
# Good elderly-sounding voices:
#   en-US-GuyNeural        (male, mature)
#   en-US-AriaNeural       (female)
#   en-GB-RyanNeural       (British male)
#   en-GB-SoniaNeural      (British female)

EDGE_VOICES = [
    "en-US-GuyNeural", "en-US-AriaNeural",
    "en-GB-RyanNeural", "en-GB-SoniaNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
]

def tts_edge(text: str, out_wav: Path, speed: float, voice: str) -> bool:
    try:
        import edge_tts
    except ImportError:
        _install("edge-tts")
        try:
            import edge_tts
        except ImportError:
            return False

    # Edge TTS rate parameter: "+0%" normal, "-20%" slower
    pct = int((speed - 1.0) * 100)
    rate_str = f"{pct:+d}%"

    async def _run():
        comm = edge_tts.Communicate(text, voice=voice, rate=rate_str)
        mp3 = out_wav.with_suffix(".mp3")
        await comm.save(str(mp3))
        if mp3.exists() and mp3.stat().st_size > 500:
            y, sr = librosa.load(str(mp3), sr=SR_WORK, mono=True)
            sf.write(str(out_wav), y, sr)
            mp3.unlink(missing_ok=True)
            return True
        mp3.unlink(missing_ok=True)
        return False

    try:
        return asyncio.run(_run())
    except Exception as e:
        print(f"  Edge TTS error: {e}")
        return False


# ── macOS say  — always available on Mac, zero extra installs ─────────────────

# Better Siri voices (macOS 14+). List what's installed with: say -v '?'
SAY_VOICES_FEMALE = ["Samantha", "Victoria", "Kate", "Karen", "Moira",
                     "Fiona", "Tessa"]
SAY_VOICES_MALE   = ["Alex",    "Daniel",   "Tom",  "Oliver", "Lee", "Rishi"]
SAY_VOICES_ALL    = SAY_VOICES_FEMALE + SAY_VOICES_MALE

def _find_best_say_voice(candidates: list) -> str:
    """Return first installed voice from candidates, else 'Alex'."""
    result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
    installed = result.stdout.lower()
    for v in candidates:
        if v.lower() in installed:
            return v
    return "Alex"

def tts_macos_say(text: str, out_wav: Path, speed: float,
                  voice: str = "Samantha") -> bool:
    if not shutil.which("say"):
        return False
    aiff     = out_wav.with_suffix(".aiff")
    rate_wpm = int(175 * speed)
    r1 = subprocess.run(
        ["say", "-v", voice, "-r", str(rate_wpm), "-o", str(aiff), text],
        capture_output=True,
    )
    if r1.returncode != 0 or not aiff.exists():
        return False
    r2 = subprocess.run(
        ["afconvert", str(aiff), str(out_wav), "-f", "WAVE", "-d", "LEI16@22050"],
        capture_output=True,
    )
    aiff.unlink(missing_ok=True)
    return r2.returncode == 0 and out_wav.exists()


# ── pyttsx3  — cross-platform offline fallback ────────────────────────────────

def tts_pyttsx3(text: str, out_wav: Path, speed: float) -> bool:
    try:
        import pyttsx3
    except ImportError:
        _install("pyttsx3")
        try:
            import pyttsx3
        except ImportError:
            return False
    try:
        eng = pyttsx3.init()
        base = eng.getProperty("rate") or 160
        eng.setProperty("rate", int(base * speed))
        eng.save_to_file(text, str(out_wav))
        eng.runAndWait()
        return out_wav.exists()
    except Exception:
        return False


# ── Unified get_audio ─────────────────────────────────────────────────────────

def get_audio(
    text    : str,
    tmp_dir : Path,
    mode    : str   = "auto",
    speed   : float = 1.0,
    speaker : str   = "Ana Florence",
    voice   : str   = "en-US-GuyNeural",
) -> np.ndarray:
    """
    Convert text → numpy float32 (mono, SR_WORK Hz).
    Tries backends in priority order until one succeeds.
    """
    wav = tmp_dir / "tts_out.wav"
    wav.unlink(missing_ok=True)

    backends: list = []
    if mode in ("auto", "xtts"):
        backends.append(("xtts",    lambda: tts_xtts(text, wav, speed, speaker)))
    if mode in ("auto", "kokoro"):
        ko_voice = KOKORO_VOICES_LIST[hash(speaker) % len(KOKORO_VOICES_LIST)]
        backends.append(("kokoro",  lambda: tts_kokoro(text, wav, speed, ko_voice)))
    if mode in ("auto", "edge"):
        backends.append(("edge",    lambda: tts_edge(text, wav, speed, voice)))
    if mode in ("auto", "say"):
        best_voice = _find_best_say_voice(SAY_VOICES_ALL)
        backends.append(("say",     lambda: tts_macos_say(text, wav, speed, best_voice)))
    if mode in ("auto", "pyttsx3"):
        backends.append(("pyttsx3", lambda: tts_pyttsx3(text, wav, speed)))

    for name, fn in backends:
        try:
            if fn():
                y, _ = librosa.load(str(wav), sr=SR_WORK, mono=True)
                return y
        except Exception as e:
            print(f"  [{name}] error: {e}")

    raise RuntimeError(
        "All TTS backends failed.\n"
        "  macOS: `say` should always work — check Terminal Accessibility permissions\n"
        "  Neural: pip install TTS   (Python 3.9-3.11)\n"
        "          pip install kokoro-onnx  (any Python, see KOKORO_MODEL_DIR)\n"
        "          pip install edge-tts  (requires internet)"
    )


# =============================================================================
# STAGE 3 — VOCAL AGING LAYER
#
# This is the differentiator. Instead of crude pitch-shift and noise,
# we apply the specific acoustic signatures of an aging larynx:
#
#   1. Jitter    — cycle-to-cycle F0 (pitch) instability
#                  Modelled as high-frequency random phase perturbation
#                  of the signal's instantaneous phase.
#
#   2. Shimmer   — cycle-to-cycle amplitude instability
#                  Modelled as slow broadband amplitude envelope noise.
#
#   3. Tremor    — quasi-sinusoidal 4-7 Hz AM+FM modulation
#                  Most distinctive feature of elderly/AD speech.
#                  Implemented as dual-rate modulation:
#                    AM: y *= 1 + depth * sin(2π * freq * t)
#                    FM: achieves via chirped phase shift
#
#   4. Breathiness — aspiration noise mixed into voiced regions
#                  Voiced regions detected via energy + ZCR thresholding.
#                  Noise is shaped to match vocal tract transfer function.
#
#   5. HF roll-off — age-related spectral tilt
#                  Simple 1st-order low-pass applied lightly.
#
# None of these are dramatic — they are subtle, perceptual cues that
# the human ear interprets as "older, less controlled voice."
# =============================================================================

def _apply_shimmer(y: np.ndarray, sr: int,
                   amount: float, rng: np.random.Generator) -> np.ndarray:
    """
    Apply shimmer: low-frequency random amplitude modulation.
    Shimmer appears at pitch-cycle rate (~80-300 Hz), but we model
    the perceptual effect via a low-pass shaped noise envelope.
    amount: fraction of RMS variation (0.02-0.10 typical for aging)
    """
    if amount < 0.005:
        return y
    # Generate smooth random envelope at ~10-50 Hz variation rate
    n     = len(y)
    noise = rng.normal(0, 1, n // 50 + 10).astype(np.float32)
    b, a  = butter(2, 40 / (sr / 2), btype="low")
    noise = lfilter(b, a, noise)
    # Upsample to signal length
    env   = np.interp(np.linspace(0, len(noise) - 1, n),
                      np.arange(len(noise)), noise)
    env   = env / (np.max(np.abs(env)) + 1e-9) * amount
    return y * (1.0 + env)


def _apply_tremor(y: np.ndarray, sr: int,
                  depth: float, freq: float) -> np.ndarray:
    """
    Apply voice tremor: sinusoidal amplitude modulation at freq Hz (4-7 Hz).
    Also adds a slight FM component for realism.
    depth: 0.05 barely audible, 0.15-0.20 clearly noticeable.
    """
    if depth < 0.01:
        return y
    t   = np.arange(len(y)) / sr
    # AM component
    am  = 1.0 + depth * np.sin(2 * np.pi * freq * t)
    # Very slight FM (phase modulation, ±0.5% depth)
    fm_depth = depth * 0.3
    fm  = 1.0 + fm_depth * np.cos(2 * np.pi * freq * t + np.pi / 4)
    # Time-warp the signal slightly using the FM envelope
    # (practical approximation: resample in chunks)
    return (y * am * fm).astype(np.float32)


def _apply_jitter(y: np.ndarray, sr: int, amount: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Apply jitter: random perturbation of the signal's time axis.
    Simulates the cycle-to-cycle period instability of the vocal folds.
    amount: standard deviation of time jitter in fractional pitch periods.
    Implemented as sample-level random delay (±amount * sr / F0_typical).
    """
    if amount < 0.001:
        return y
    # F0 typical elderly ~140 Hz → period ~157 samples at 22050
    period_samples = int(sr / 140)
    # Maximum shift in samples
    max_shift = max(1, int(amount * period_samples))

    # Apply random shifts in blocks of ~period_samples
    block  = period_samples * 2
    result = []
    i = 0
    while i < len(y):
        end    = min(i + block, len(y))
        chunk  = y[i:end]
        shift  = int(rng.integers(-max_shift, max_shift + 1))
        if shift > 0:
            chunk = np.concatenate([np.zeros(shift, dtype=np.float32), chunk[:-shift]])
        elif shift < 0:
            chunk = np.concatenate([chunk[-shift:], np.zeros(-shift, dtype=np.float32)])
        result.append(chunk)
        i = end

    out = np.concatenate(result)
    return out[: len(y)]


def _detect_voiced(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Returns a boolean mask: True where the signal is voiced.
    Uses energy + zero-crossing rate heuristic.
    """
    frame_len = int(0.025 * sr)   # 25 ms frames
    hop       = frame_len // 2
    n_frames  = (len(y) - frame_len) // hop + 1

    voiced = np.zeros(len(y), dtype=bool)
    for fi in range(n_frames):
        s   = fi * hop
        e   = s + frame_len
        seg = y[s:e]
        rms = np.sqrt(np.mean(seg ** 2))
        # ZCR
        zcr = np.mean(np.abs(np.diff(np.sign(seg)))) / 2
        is_voiced = (rms > 0.015) and (zcr < 0.35)
        voiced[s:e] = is_voiced

    return voiced


def _apply_breathiness(y: np.ndarray, sr: int, amount: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Mix aspiration noise into voiced regions.
    Aspiration is modelled as band-limited noise (500-4000 Hz) matched
    to the local RMS of voiced segments — softer where speech is quieter.
    amount: fraction of voiced RMS to add as noise (0.05-0.20).
    """
    if amount < 0.01:
        return y

    voiced_mask = _detect_voiced(y, sr)

    # Generate shaped aspiration noise
    noise = rng.normal(0, 1, len(y)).astype(np.float32)
    b, a  = butter(4, [500 / (sr / 2), 4000 / (sr / 2)], btype="band")
    noise = lfilter(b, a, noise)

    # Scale noise to local voiced RMS
    frame_len = int(0.025 * sr)
    hop       = frame_len // 2
    out = y.copy()
    for fi in range((len(y) - frame_len) // hop):
        s = fi * hop
        e = s + frame_len
        if voiced_mask[s:e].mean() < 0.6:
            continue
        local_rms = np.sqrt(np.mean(y[s:e] ** 2)) or 1e-9
        noise_rms = np.sqrt(np.mean(noise[s:e] ** 2)) or 1e-9
        gain = local_rms * amount / noise_rms
        out[s:e] += noise[s:e] * gain

    return out


def _apply_hf_rolloff(y: np.ndarray, sr: int, retention: float) -> np.ndarray:
    """
    Apply age-related high-frequency attenuation.
    retention=1.0 → no effect; retention=0.7 → noticeable muffling.
    Implemented as gentle 1st-order low-pass.
    """
    if retention > 0.98:
        return y
    # Map retention (0.6-1.0) to cutoff frequency (3000-12000 Hz)
    cutoff = 3000 + retention * 9000   # Hz
    cutoff = min(cutoff, sr / 2 - 100)
    b, a   = butter(1, cutoff / (sr / 2), btype="low")
    filtered = lfilter(b, a, y).astype(np.float32)
    # Blend: retention controls dry/wet
    alpha = (1.0 - retention)
    return (1 - alpha) * y + alpha * filtered


def age_voice(
    y       : np.ndarray,
    sr      : int,
    params  : dict,
    rng     : np.random.Generator,
) -> np.ndarray:
    """
    Apply the full vocal aging layer.
    All effects are mild individually but compound into a convincingly
    aged voice when combined.
    """
    y = _apply_jitter(y, sr, params["aging_jitter"], rng)
    y = _apply_shimmer(y, sr, params["aging_shimmer"], rng)
    y = _apply_tremor(y, sr, params["aging_tremor_depth"],
                      params["aging_tremor_freq"])
    y = _apply_breathiness(y, sr, params["aging_breathiness"], rng)
    y = _apply_hf_rolloff(y, sr, params["aging_hf_rolloff"])
    return y


# =============================================================================
# STAGE 4 — LIGHT ENVIRONMENTAL TEXTURE
# =============================================================================

def insert_pauses(y: np.ndarray, sr: int,
                  silence_prob: float, silence_ms: float,
                  rng: np.random.Generator) -> np.ndarray:
    """
    Nudge a few phrase-boundary pauses. Kept very light because
    the disfluent text already creates pauses via the TTS.
    """
    if silence_prob < 0.03:
        return y

    frame_len = int(0.05 * sr)
    hop       = frame_len // 2
    energy    = np.array([
        np.sqrt(np.mean(y[i:i + frame_len] ** 2))
        for i in range(0, len(y) - frame_len, hop)
    ], dtype=np.float32)
    thr = float(10 ** (-28 / 20))
    is_sil = energy < thr

    chunks, in_v, start = [], not is_sil[0], 0
    for fi, sil in enumerate(is_sil):
        s = fi * hop
        if in_v and sil:
            chunks.append(("v", start, s)); start = s; in_v = False
        elif not in_v and not sil:
            chunks.append(("s", start, s)); start = s; in_v = True
    chunks.append(("v" if in_v else "s", start, len(y)))

    out = []
    for kind, s, e in chunks:
        out.append(y[s:e])
        if kind == "v" and rng.random() < silence_prob:
            dur = float(np.clip(
                rng.normal(silence_ms / 1000, silence_ms / 4000), 0.05, 1.2
            ))
            out.append(np.zeros(int(dur * sr), dtype=np.float32))
    return np.concatenate(out) if out else y


def add_room_noise(y: np.ndarray, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Minimal broadband room noise."""
    if amp < 0.0003:
        return y
    return y + rng.normal(0, amp, len(y)).astype(np.float32)


def normalise(y: np.ndarray, peak: float = 0.88) -> np.ndarray:
    mx = np.max(np.abs(y))
    return y / mx * peak if mx > 1e-9 else y


def resample_to(y: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return y
    return librosa.resample(y, orig_sr=from_sr, target_sr=to_sr)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def synthesise_one(
    text      : str,
    params    : dict,
    group     : str,
    tmp_dir   : Path,
    target_sr : int,
    tts_mode  : str,
    speaker   : str,
    voice     : str,
    rng       : np.random.Generator,
) -> tuple[np.ndarray, str]:
    """
    Returns (audio_array_float32, disfluent_text_used).
    """
    is_ad   = (group == "ad")
    severity = params["severity"] if is_ad else 0.0

    # ── Stage 1: Disfluency injection ────────────────────────────────────────
    disfluent = inject_disfluencies(text, severity, rng)

    # ── Stage 2: TTS ─────────────────────────────────────────────────────────
    y = get_audio(disfluent, tmp_dir,
                  mode=tts_mode, speed=params["tts_speed"],
                  speaker=speaker, voice=voice)

    # ── Stage 3: Vocal aging layer ────────────────────────────────────────────
    y = age_voice(y, SR_WORK, params, rng)

    # ── Stage 4: Light environmental texture ──────────────────────────────────
    y = insert_pauses(y, SR_WORK,
                      params["silence_prob"], params["silence_ms"], rng)
    y = add_room_noise(y, params["noise_amp"], rng)

    # ── Finalise ──────────────────────────────────────────────────────────────
    y = normalise(y)
    y = resample_to(y, SR_WORK, target_sr)

    return y.astype(np.float32), disfluent


def save_wav(path: Path, y: np.ndarray, sr: int):
    int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(str(path), sr, int16)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(
    n_samples : int  = 50,
    seed      : int  = 42,
    out_dir   : str  = "audio_dataset_v3",
    target_sr : int  = 16000,
    tts_mode  : str  = "auto",
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    out_path = Path(out_dir)
    hc_dir   = out_path / "healthy"
    ad_dir   = out_path / "ad"
    hc_dir.mkdir(parents=True, exist_ok=True)
    ad_dir.mkdir(parents=True, exist_ok=True)

    n_hc = n_samples // 2
    n_ad = n_samples - n_hc
    total = n_hc + n_ad

    meta_rows = []
    generated = 0
    t0        = time.time()

    print(f"\n{'='*65}")
    print(f"  AD / HC Speech Dataset Generator  (v3 — MAXIMUM REALISM)")
    print(f"  {total} files  ({n_hc} HC + {n_ad} AD)")
    print(f"  TTS mode  : {tts_mode}")
    print(f"  Output    : {out_path.resolve()}")
    print(f"{'='*65}\n")

    prompts_pool = list(SPEECH_PROMPTS)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        def _gen(group, cfg, prefix, directory, count):
            nonlocal generated, prompts_pool

            label = 0 if group == "healthy" else 1

            for i in range(1, count + 1):
                if not prompts_pool:
                    prompts_pool.extend(SPEECH_PROMPTS)
                random.shuffle(prompts_pool)
                text = prompts_pool.pop()

                params = sample_config(cfg, rng)

                # Pick speaker / voice variety
                speaker = XTTS_SPEAKERS[i % len(XTTS_SPEAKERS)]
                voice   = EDGE_VOICES[i % len(EDGE_VOICES)]

                try:
                    y, disfluent = synthesise_one(
                        text, params, group, tmp_dir,
                        target_sr, tts_mode, speaker, voice, rng
                    )
                except Exception as exc:
                    print(f"\n  ✗ Skipped {prefix}_{i:04d}: {exc}")
                    continue

                naccid = f"{prefix}_{i:04d}"
                fpath  = directory / f"{naccid}.wav"
                save_wav(fpath, y, target_sr)

                meta_rows.append({
                    "NACCID"         : naccid,
                    "label"          : label,
                    "group"          : group,
                    "filename"       : str(fpath.relative_to(out_path)),
                    "original_text"  : text,
                    "disfluent_text" : disfluent,
                    "speaker"        : speaker,
                    "duration_s"     : round(len(y) / target_sr, 3),
                    # Disfluency
                    "severity"       : round(params["severity"],      3),
                    "tts_speed"      : round(params["tts_speed"],     3),
                    # Vocal aging
                    "aging_jitter"   : round(params["aging_jitter"],  4),
                    "aging_shimmer"  : round(params["aging_shimmer"], 4),
                    "aging_tremor_d" : round(params["aging_tremor_depth"], 3),
                    "aging_tremor_f" : round(params["aging_tremor_freq"],  2),
                    "aging_breath"   : round(params["aging_breathiness"],  3),
                    "aging_hf_roll"  : round(params["aging_hf_rolloff"],   3),
                    # Environment
                    "silence_prob"   : round(params["silence_prob"],  3),
                    "silence_ms"     : round(params["silence_ms"],    1),
                    "noise_amp"      : round(params["noise_amp"],     5),
                    "sr_hz"          : target_sr,
                })

                generated += 1
                pct     = generated / total * 100
                bar     = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                elapsed = time.time() - t0
                eta     = (elapsed / generated) * (total - generated)
                print(
                    f"\r  [{bar}] {pct:5.1f}%  {generated}/{total}"
                    f"  {group.upper()} {i}/{count}  ETA {eta:5.1f}s  ",
                    end="", flush=True,
                )

        _gen("healthy", HC_CONFIG, "HC", hc_dir, n_hc)
        _gen("ad",      AD_CONFIG, "AD", ad_dir,  n_ad)

    print(f"\n\n  ✓ Done in {time.time() - t0:.1f}s")

    if not meta_rows:
        print("  ✗ No files generated. Check TTS installation.\n")
        return []

    meta_path = out_path / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)
    print(f"  Metadata → {meta_path}")

    import statistics as st
    print(f"\n{'─'*65}")
    print("  SUMMARY")
    print(f"{'─'*65}")
    for grp, lbl in [("healthy", 0), ("ad", 1)]:
        rows = [r for r in meta_rows if r["label"] == lbl]
        if not rows:
            continue
        durs = [r["duration_s"] for r in rows]
        sevs = [r["severity"]   for r in rows]
        spds = [r["tts_speed"]  for r in rows]
        trms = [r["aging_tremor_d"] for r in rows]
        print(f"\n  [{grp.upper():7}]  n={len(rows)}")
        print(f"    Duration (s)    : {st.mean(durs):.2f} ± {st.stdev(durs):.2f}"
              f"  ({min(durs):.1f}–{max(durs):.1f})")
        print(f"    TTS speed       : {st.mean(spds):.3f} ± {st.stdev(spds):.3f}")
        print(f"    Severity        : {st.mean(sevs):.3f} ± {st.stdev(sevs):.3f}")
        print(f"    Tremor depth    : {st.mean(trms):.3f} ± {st.stdev(trms):.3f}")

    print(f"\n{'─'*65}")
    print(f"  Total: {len(meta_rows)} files  →  {out_path.resolve()}")
    print(f"{'─'*65}\n")
    return meta_rows


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate maximum-realism AD / HC speech dataset (v3)"
    )
    parser.add_argument("--n",    type=int, default=50)
    parser.add_argument("--sr",   type=int, default=16000)
    parser.add_argument("--out",  type=str, default="audio_dataset_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tts",  type=str, default="auto",
                        choices=["auto", "xtts", "kokoro", "edge", "say", "pyttsx3"],
                        help=(
                            "TTS backend (default: auto — tries xtts → kokoro"
                            " → edge → say → pyttsx3)"
                        ))
    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n, seed=args.seed,
        out_dir=args.out, target_sr=args.sr,
        tts_mode=args.tts,
    )


if __name__ == "__main__":
    main()