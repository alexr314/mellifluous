"""Phase-vocoder time-stretch: change duration without changing pitch.

The classic algorithm (Flanagan and Golden 1966, popularized by Dolson 1986):

  1. Take the STFT of the input with hop H_a (analysis hop).
  2. For each desired output frame at hop H_s (synthesis hop = H_a / speed),
     compute the cumulative phase by advancing the previous output phase by
     the "true frequency" implied by the phase difference between adjacent
     input frames -- thereby preserving the harmonic structure.
  3. Inverse-STFT the resulting frames.

For speech in the 0.5x-2x range this sounds clean; beyond that you get
mild "phasiness" but it's still intelligible. We use a 50%-overlap Hann
window and the standard fft_size = 2048 for 24 kHz speech.

scipy is required (already a transitive dep of mlx-audio / soundfile, so
no new install). No librosa, no numba.
"""
from __future__ import annotations
import numpy as np


_FFT_SIZE = 2048               # ~85 ms at 24 kHz; standard for speech
_ANALYSIS_HOP = _FFT_SIZE // 4 # 75% overlap -- balances quality and cost


def time_stretch(pcm: np.ndarray, speed: float) -> np.ndarray:
    """Return `pcm` resampled to play `speed`x as fast (1.5 = 1.5x faster /
    66.6% of original duration), with pitch preserved.

    `pcm` is float32 mono in [-1, 1]. Returns float32 mono. speed=1.0 is a
    pass-through (returns the input unchanged).
    """
    if speed == 1.0:
        return pcm
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")
    if pcm.size == 0:
        return pcm

    # Synthesis hop is shorter than analysis hop when speeding up. The
    # ratio of hops controls how many output frames we generate for each
    # input frame -- which is exactly the stretch factor.
    synth_hop = max(1, int(round(_ANALYSIS_HOP / speed)))
    analysis_hop = _ANALYSIS_HOP

    # Hann window. The 50%-overlap variant (with fft_size/4 hop = 75%
    # overlap) is COLA-compliant, so iSTFT recovers the signal cleanly.
    window = np.hanning(_FFT_SIZE).astype(np.float32)

    # Pad so the first and last samples sit in the middle of the first
    # and last analysis frames.
    pad = _FFT_SIZE // 2
    padded = np.pad(pcm, (pad, pad), mode="constant")

    # Analysis STFT. We compute frames at every `analysis_hop` samples.
    n_input_frames = 1 + (len(padded) - _FFT_SIZE) // analysis_hop
    if n_input_frames <= 1:
        return pcm

    input_frames = np.empty((n_input_frames, _FFT_SIZE), dtype=np.float32)
    for i in range(n_input_frames):
        start = i * analysis_hop
        input_frames[i] = padded[start:start + _FFT_SIZE] * window
    input_spec = np.fft.rfft(input_frames, axis=1)

    input_mag   = np.abs(input_spec).astype(np.float32)
    input_phase = np.angle(input_spec).astype(np.float32)

    # Expected phase advance per analysis hop for each frequency bin
    # (assuming a stationary sinusoid in that bin). Used to unwrap the
    # frame-to-frame phase difference into a "true" frequency offset.
    n_bins = input_spec.shape[1]
    bin_phase_advance = (
        2.0 * np.pi * analysis_hop * np.arange(n_bins, dtype=np.float32) / _FFT_SIZE
    )

    # Stretch ratio expressed in frames (not samples): each output frame
    # advances `1/speed` input frames in time. Output length in frames:
    n_output_frames = max(1, int(round(n_input_frames / speed)))

    # Accumulate output phase frame-by-frame so spectral peaks stay
    # coherent across the time-stretched signal.
    output_phase = input_phase[0].copy()
    output_spec = np.empty((n_output_frames, n_bins), dtype=np.complex64)
    output_spec[0] = input_mag[0] * np.exp(1j * output_phase)

    for j in range(1, n_output_frames):
        # Where this output frame "comes from" in input-frame coordinates.
        # Linear interpolation between two adjacent input frames keeps the
        # magnitude envelope smooth.
        src = j * (n_input_frames - 1) / (n_output_frames - 1) if n_output_frames > 1 else 0
        i0 = int(np.floor(src))
        i1 = min(i0 + 1, n_input_frames - 1)
        frac = src - i0

        mag = (1 - frac) * input_mag[i0] + frac * input_mag[i1]

        # True frequency: unwrap the phase difference between the two
        # adjacent input frames and remove the expected advance for each
        # bin. What's left is the deviation, which we re-add at the
        # synthesis hop scale.
        dphi = input_phase[i1] - input_phase[i0] - bin_phase_advance
        # Wrap to (-pi, pi]
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        true_freq = bin_phase_advance + dphi
        output_phase = output_phase + true_freq * (synth_hop / analysis_hop)
        output_spec[j] = mag * np.exp(1j * output_phase)

    # Inverse STFT via overlap-add. Each output frame is windowed and
    # added into the output buffer at offset j * synth_hop.
    out_len = (n_output_frames - 1) * synth_hop + _FFT_SIZE
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)  # for normalization
    for j in range(n_output_frames):
        frame = np.fft.irfft(output_spec[j], n=_FFT_SIZE).astype(np.float32)
        start = j * synth_hop
        out[start:start + _FFT_SIZE] += frame * window
        wsum[start:start + _FFT_SIZE] += window * window
    # Normalize by the window squared sum to undo COLA scaling.
    out = np.divide(out, wsum, out=out, where=wsum > 1e-6)

    # Trim the leading and trailing pad regions to align with the input.
    out = out[pad:len(out) - pad]
    # Match expected output length (round-off can leave us 1-2 samples off).
    target = max(1, int(round(len(pcm) / speed)))
    if len(out) > target:
        out = out[:target]
    elif len(out) < target:
        out = np.pad(out, (0, target - len(out)), mode="constant")
    return out.astype(np.float32, copy=False)
