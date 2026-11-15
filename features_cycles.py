"""The 'Laplace equation' as a PDE does not apply to 1D price.
This module implements the legitimate spectral / analytic-signal equivalent.
Only FFT / Hilbert features are returned.
The Laplace PDE does not apply to 1D price. This is the correct spectral replacement.
"""
import numpy as np
from features_pro import _prep

KEYS = ("fft_dominant_period_bars", "fft_cycle_strength", "hilbert_phase_deg",
        "hilbert_amplitude_pct", "hilbert_cycle_direction")


def compute(df, side=None):
    d = _prep(df)
    if d is None or len(d)<128:
        return dict.fromkeys(KEYS)
    c = d.close.to_numpy()[-128:]
    spectrum = np.fft.fft(c-c.mean())
    power = abs(spectrum[1:65])**2
    if power.max() == 0 or c[-1]<=0:
        return dict.fromkeys(KEYS)
    peak = int(power.argmax())
    spectrum[1:64] *= 2
    spectrum[65:] = 0
    analytic = np.fft.ifft(spectrum)
    phase = np.unwrap(np.angle(analytic[-2:]))
    return dict(zip(KEYS,(128/(peak+1), float(power[peak]/power.mean()),float(np.degrees(phase[-1])%360),
                          float(100*abs(analytic[-1])/c[-1]), "UP" if phase[-1]>=phase[-2] else "DOWN")))
